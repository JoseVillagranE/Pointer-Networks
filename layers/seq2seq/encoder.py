# coding=utf-8

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable

def rnn_factory(rnn_type, **kwargs):
  pack_padded_seq = True
  if rnn_type in ["LSTM", "GRU", "RNN"]:
    rnn = getattr(nn, rnn_type)(**kwargs) # Call the method of torch for LSTM, GRU or RNN
  return rnn, pack_padded_seq

class EncoderBase(nn.Module):
  """ encoder base class
  """
  def __init__(self):
    super(EncoderBase, self).__init__()

  def forward(self, src, lengths=None, hidden=None):
    """
    Args:
      src (FloatTensor) : input sequence 
      lengths (LongTensor) : lengths of input sequence
      hidden : init hidden state
    """
    raise NotImplementedError()

class RNNEncoder(EncoderBase):
  """ RNN encoder class

  Args:
    rnn_type : rnn cell type, ["LSTM", "GRU", "RNN"]
    bidirectional : whether use bidirectional rnn
    num_layers : number of layers in stacked rnn
    input_size : input dimension size
    hidden_size : rnn hidden dimension size
    dropout : dropout rate
    use_bridge : TODO: implement bridge
  """
  def __init__(self, rnn_type, bidirectional, num_layers,
    input_size, hidden_size, dropout, use_bridge=False):
    super(RNNEncoder, self).__init__()
    if bidirectional:
      assert hidden_size % 2 == 0
      # hidden_size = hidden_size // 2
    self.rnn, self.pack_padded_seq = rnn_factory(rnn_type,
      input_size=input_size,
      hidden_size=hidden_size,
      bidirectional=bidirectional,
      num_layers=num_layers,
      dropout=dropout)
    self.use_bridge = use_bridge
    self.bidirectional = bidirectional
    self.enc_init_state = self.init_hidden(hidden_size)
    
    if self.use_bridge:
      raise NotImplementedError()
  
  def forward(self, src, lengths=None, hidden=None):
    """
    Same as BaseEncoder.forward
    """
    packed_src = src
    if self.pack_padded_seq and lengths is not None:
      lengths = lengths.view(-1).tolist()
      packed_src = pack(src, lengths)
      
      
    h_0 = self.enc_init_state[0].repeat(src.shape[1], 1).unsqueeze(0)
    c_0 = self.enc_init_state[1].repeat(src.shape[1], 1).unsqueeze(0)
    
    if self.bidirectional:
        h_0 = h_0.repeat(2, 1, 1)
        c_0 = c_0.repeat(2, 1, 1)
    memory_bank, hidden_final = self.rnn(packed_src,(h_0, c_0))

    if self.pack_padded_seq and lengths is not None:
      memory_bank = unpack(memory_bank)[0]
    
    if self.use_bridge:
      raise NotImplementedError()
    return memory_bank, hidden_final

  def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        enc_init_hx = Variable(torch.zeros(hidden_dim), requires_grad=True)
        enc_init_cx = Variable(torch.zeros(hidden_dim), requires_grad=True)
        if torch.cuda.is_available():
            enc_init_cx = enc_init_cx.cuda()
            enc_init_hx = enc_init_hx.cuda()
        return (enc_init_hx, enc_init_cx)  