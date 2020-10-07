# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

def apply_mask(align_score, mask, prev_idxs):
    """ apply mask for shutdown previous indexs that already chose
    Args:
    align_score : scores
    mask : mask for content indexs with booleans
    prev_idxs: Previous indexs that already the algorithm chose
    Return:
    align_score
    """
    if mask is None:
        mask = torch.zeros(align_score.size()).byte() # Byte Tensor
        if torch.cuda.is_available():
            mask = mask.cuda()
    
    mask_ = mask.clone()
    if prev_idxs is not None:
        # [prev_idxs[i] for i in range(prev_idxs.shape[0])]
        mask_[[x for x in range(align_score.size(0))],:, prev_idxs.data] = 1
        align_score[mask_] = -np.inf
    return align_score, mask_
class Attention(nn.Module):
    """ Attention layer
    Args:
      dim : hidden dimension size
    """
    def __init__(self, hidden_dim, mask_bool=False, hidden_att_bool=False,
                 C=None, is_cuda_available=False):
        super().__init__()
        self.mask_bool = mask_bool
        self.hidden_att_bool = hidden_att_bool
        self.C = C
        self.tanh = nn.Tanh()
        self.W_ref = nn.Linear(hidden_dim, hidden_dim, bias=False) # En el paper es matriz. Revisar!
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        if is_cuda_available:
            self.W_ref = self.W_ref.cuda()
            self.W_q = self.W_q.cuda()
            self.v = self.v.cuda()
    def forward(self, src, tgt, mask=None, prev_idxs=None):
        """
        Args:
        src : source values (bz, seq_len, hidden_dim). Hidden state of encoder
        
        tgt : target values (bz, 1, hidden_dim). dec_i or q 
        src_lengths : source values length
        """

        temp = self.tanh(self.W_q(tgt) + self.W_ref(src))
        u = self.v(temp).transpose(1, 2)
        if self.C:
            logit = self.C*self.tanh(u)
        else:
            logit = u
        
        if self.mask_bool: 
            logit, mask = apply_mask(logit, mask, prev_idxs)
        # Normalize weights
        # print(logit)
        probs = F.softmax(logit, -1) # [batch, 1, seq_len]
        
        if self.hidden_att_bool:
            # probs = probs.transpose(1,2)
            # d = probs*src # pointer network paper
            d_prime= torch.bmm(probs, src)
            # d = d.sum(dim=2)
            # concat_d = torch.cat([d.unsqueeze(1), tgt], -1)
            concat_d = (d_prime.transpose(0, 1), tgt.transpose(0, 1))
            
            return concat_d, probs, logit, mask
        else:
            return probs, logit, mask # [batch_size, hidden_dim, embedding_dim], [batch_size, 1, embedding_dim]
    
    
if __name__ == "__main__":
    
    attn = Attention("general", 64, 10)
    
    inp = torch.randn((10, 2, 64))
    tgt = torch.randn(10, 1, 64)
    
    _, probs, _ = attn(inp, tgt)
    print(probs.shape)
    