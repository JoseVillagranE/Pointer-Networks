# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

def apply_mask(align_score, mask, prev_idxs=None):
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
        mask_[[x for x in range(align_score.size(0))], prev_idxs.data] = 1
        align_score[mask_] = -np.inf
        
    return align_score, mask_
    
class Attention(nn.Module):
    """ Attention layer
    Args:
      dim : hidden dimension size
    """
    def __init__(self, hidden_dim, mask_bool=False, hidden_att_bool=False,
                 C=None, T=1, is_cuda_available=False):
        super().__init__()
        self.mask_bool = mask_bool
        self.hidden_att_bool = hidden_att_bool
        self.C = C
        self.T = T
        self.tanh = nn.Tanh()
        
        self.W_ref = nn.Conv1d(hidden_dim, hidden_dim, 1, 1)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        if is_cuda_available:
            self.W_ref = self.W_ref.cuda()
            self.W_q = self.W_q.cuda()
            self.V = nn.Parameter(torch.cuda.FloatTensor(hidden_dim))
        
        else:
            self.V = nn.Parameter(torch.FloatTensor(hidden_dim)).cuda()
            
            
    def forward(self, src, tgt, mask=None, prev_idxs=None):
        """
        Args:
        src : source values (bz, seq_len, hidden_dim). Hidden state of encoder
        
        tgt : target values (bz, hidden_dim). dec_i or q 
        src_lengths : source values length
        """
        u1 = self.W_q(tgt).unsqueeze(-1).repeat(1, 1, src.shape[1])
        u2 = self.W_ref(src.permute(0, 2, 1)) # [Batch, hidden_dim, seq_len]
        V = self.V.unsqueeze(0).unsqueeze(0).repeat(src.shape[0], 1, 1)
        u = torch.bmm(V, self.tanh(u1 + u2)).squeeze(1) # [Batch, 1, hidden] x [Batch, hidden_dim, seq_len]
        
        if self.C:
            logit = self.C*self.tanh(u)
        else:
            logit = u
        if self.mask_bool: 
            logit, mask = apply_mask(logit, mask, prev_idxs)
        # Normalize weights
        probs = F.softmax(logit/self.T, -1) # [batch, seq_len]
        d_prime= torch.bmm(u2, probs.unsqueeze(-1)).squeeze(-1) # [batch, hidden_size]
        if self.hidden_att_bool:
            concat_d = (d_prime.transpose(0, 1), tgt.transpose(0, 1))
            
            return concat_d, probs, logit, mask
        else:
            return d_prime, probs.squeeze(), logit, mask
    
    
if __name__ == "__main__":
    
    attn = Attention("general", 64, 10)
    
    inp = torch.randn((10, 2, 64))
    tgt = torch.randn(10, 1, 64)
    
    _, probs, _ = attn(inp, tgt)
    print(probs.shape)
    