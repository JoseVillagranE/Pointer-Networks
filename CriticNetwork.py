# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:24:00 2020

@author: joser
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


from layers.seq2seq.encoder import RNNEncoder
from layers.seq2seq.decoder import RNNDecoderBase
from layers.attention import Attention, sequence_mask
from torch import optim

def GlimpseFunction(ref, attn_prob):
    """
    input: 
        ref: salida del encoder(ref -> Bello)
        atten_prob: probabilidades (salida del mecanismo de atención)
    output:
        outp: salida de la función Glimpse
    """
    ref = ref.transpose(2, 1)
    sm = nn.Softmax()
    outp = torch.bmm(sm(attn_prob), ref)
    return outp
    

class CriticNetwork(nn.Module):
    
    def __init__(self, rnn_type, num_layers, bidirectional, embedding_dim, hidden_dim, 
                 process_block_iter, batch_size, dropout=0.0, C=None, is_cuda_available=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.process_block_iter = process_block_iter
#        self.inp_len = inp_len
        
        self.encoder = RNNEncoder(rnn_type, bidirectional, num_layers, embedding_dim,
                                  hidden_dim, dropout)
        self.process_block = Attention("RL", hidden_dim, batch_size, C=C)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))
        
        self.sm = nn.Softmax()
        
        self.is_cuda_available = is_cuda_available
        
        
        if is_cuda_available:
            self.encoder = self.encoder.cuda()
            self.process_block = self.process_block.cuda()
            self.decoder = self.decoder.cuda()
            
        
        
        
    def forward(self, inp, inp_len):
        
        '''
        input:
            inp: Muestras de Tour
            inp_len: cantidad de coordenas más una variable dummy
        output:
            outp: Baseline calculado por el critic.
        
        '''
        inp = inp.transpose(0, 1)
        memory_bank, (hidden, c_n) = self.encoder(inp, inp_len)
        memory_bank = memory_bank.transpose(0, 1) # [batch_size, emb_size, emb_size]
        hidden = hidden.transpose(0, 1) # [batch_size, 1, hidden_size]
        for i in range(self.process_block_iter):
            attn_h, align_score = self.process_block(memory_bank, hidden) # modifica las dimensiones de la matriz. Se debe verificar
            hidden = GlimpseFunction(attn_h, align_score)
        
        outp = self.decoder(hidden)
        
        return outp
        
        