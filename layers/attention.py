# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F

def sequence_mask(lengths, max_len=None):
  """ Crete mask for lengths
  Args:
    lengths (LongTensor) : lengths (bz)
    max_len (int) : maximum length
  Return:
    mask (bz, max_len)
  """
  bz = lengths.numel()
  max_len = max_len or lengths.max()
  return (torch.arange(0, max_len)
        .type_as(lengths)
        .repeat(bz, 1)
        .lt(lengths))

class Attention(nn.Module):
  """ Attention layer
  Args:
    attn_type : attention type ["dot", "general"]
    dim : hidden dimension size
  """
  def __init__(self, attn_type, dim, bz_size, C=None):
    super().__init__()
    self.attn_type = attn_type
    self.C = C
    
    bias_out = attn_type == "mlp"
    self.linear_out = nn.Linear(dim *2, dim, bias_out)
    self.conv_proj = nn.Conv1d(dim, dim, 1 , 1)
    self.v = 0
    if self.attn_type == "RL":
        self.W_ref = nn.Linear(dim, dim, bias=False)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.v = torch.FloatTensor(torch.ones((bz_size, 1, dim))).cuda()
    elif self.attn_type == "general":
        self.linear = nn.Linear(dim, dim, bias=False)
    elif self.attn_type == "dot":
      pass
    else:
      raise NotImplementedError()
  
  def score(self, src, tgt):
    """ Attention score calculation
    Args:
      src : source values (bz, src_len, dim)
      tgt : target values (bz, tgt_len, dim)
    """
    # bz, src_len, dim = src.size()
    # _, tgt_len, _ = tgt.size()

    if self.attn_type in ["general", "dot", "RL"]:
      tgt_ = tgt
      src_ = src
      if self.attn_type == "RL":
            tgt_ = self.W_q(tgt_)
            src_ = self.W_ref(src)
      elif self.attn_type == "general":
          tgt_ = self.linear(tgt_)
      src_ = src_.transpose(1, 2)
      
      if self.attn_type in ["general", "dot"]:
          return torch.bmm(tgt_, src_)
      elif type(self.v)=='torch.Tensor':
          u = torch.bmm(self.v, torch.tanh(torch.bmm(tgt_, src_)))
          if C:
              return self.C*torch.tanh(u)
          else:
              return u
      else:
          return torch.tanh(torch.bmm(tgt_, src_))
    else:
      raise NotImplementedError()
  
  def forward(self, src, tgt, src_lengths=None):
    """
    Args:
      src : source values (bz, src_len, dim). enc_i or ref in Bello's Paper
      tgt : target values (bz, tgt_len, dim). dec_i or q
      src_lengths : source values length
    """
    if tgt.dim() == 2:
      one_step = True
      src = src.unsqueeze(1)
    else:
      one_step = False
    
    # bz, src_len, dim = src.size()
    # _, tgt_len, _ = tgt.size()

    align_score = self.score(src, tgt)

    if src_lengths is not None:
      mask = sequence_mask(src_lengths)
      # (bz, max_len) -> (bz, 1, max_len)
      # so mask can broadcast
      mask = mask.unsqueeze(1)
      align_score.data.masked_fill_(~mask, -float('inf'))
    # align_score = align_score.squeeze()
    # Normalize weights
    align_score = F.softmax(align_score.squeeze(), -1)

    align_score = align_score.unsqueeze(2).transpose(1,2)
    attn_h = 0
    if self.attn_type in ["general", "dot"]:
        c = torch.bmm(align_score, src)
        concat_c = torch.cat([c, tgt], -1)
        attn_h = self.linear_out(concat_c)
        if one_step:
          attn_h = attn_h.squeeze(1)
          align_score = align_score.squeeze(1)
    else:
        src = src.transpose(1, 2)
        attn_h = self.conv_proj(src)
    return attn_h, align_score # [batch_size, hidden_dim, embedding_dim], [batch_size, 1, embedding_dim]