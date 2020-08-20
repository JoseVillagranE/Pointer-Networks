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
        mask_[[x for x in range(align_score.size(0))],:, prev_idxs] = 1
        align_score[mask_] = -np.inf
    return align_score, mask_
class Attention(nn.Module):
    """ Attention layer
    Args:
      attn_type : attention type ["dot", "general"]
      dim : hidden dimension size
    """
    def __init__(self, attn_type, dim, bz_size, C=None, is_cuda_available=False):
        super().__init__()
        self.C = C
        self.tanh = nn.Tanh()
        self.conv_proj = nn.Conv1d(dim, dim, 1 , 1)
        self.W_ref = nn.Linear(dim, dim, bias=False) # En el paper es matriz. Revisar!
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.v = torch.FloatTensor(dim)
        if is_cuda_available:
            self.W_ref = self.W_ref.cuda()
            self.W_q = self.W_q.cuda()
            self.conv_proj = self.conv_proj.cuda()
            self.v = self.v.cuda()
        
        self.v.data.uniform_(0, 1)
        self.v = nn.Parameter(self.v)
    def forward(self, src, tgt, mask=None, prev_idxs=None, training_type = "RL", attention_type="Attention"):
        """
        Args:
        src : source values (bz, src_len, hidden_dim). enc_i or ref in Bello's Paper
        
        tgt : target values (bz, 1, hidden_dim). dec_i or q
        src_lengths : source values length
        """
        
        
        v = self.v.unsqueeze(0).expand(src.size(0), len(self.v)).unsqueeze(1)
        # [batch, 1, hidden_dim] x [batch, hidden, seq_len]
        u = torch.bmm(v,self.tanh(self.W_q(tgt) + self.W_ref(src)).transpose(1, 2))#.transpose(1, 2)
        if self.C:
            logit = self.C*self.tanh(u)
        else:
            logit = u
        
        if attention_type == Attention and training_type == "RL": 
            logit, mask = apply_mask(logit, mask, prev_idxs)
        # Normalize weights
        probs = F.softmax(logit, -1) # [batch, 1, seq_len]
        
        # if len(probs.size())!=1:
        #     probs = probs.transpose(1,2)
          
        if training_type == "Sup":
            # probs = probs.transpose(1,2)
            # d = probs*src # pointer network paper
            d_prime= torch.bmm(probs, src)
            # d = d.sum(dim=2)
            # concat_d = torch.cat([d.unsqueeze(1), tgt], -1)
            concat_d = (d_prime.transpose(0, 1), tgt.transpose(0, 1))
            
            return concat_d, probs, logit
        # if one_step:
        #     attn_h = attn_h.squeeze(1)
        #     probs = probs.squeeze(1)
        # else:
        return concat_d, probs, mask # [batch_size, hidden_dim, embedding_dim], [batch_size, 1, embedding_dim]