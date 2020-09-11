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
from layers.attention import Attention
from torch import optim

    

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
        # self.process_block = RNNEncoder(rnn_type, bidirectional, num_layers, embedding_dim,
        #                           hidden_dim, dropout)
        self.process_block = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        self.attending = Attention("RL", hidden_dim, batch_size, C=None)
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
        
        (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        encoder_hx = encoder_hx.unsqueeze(0).repeat(inp.size(1), 1).unsqueeze(0)       
        encoder_cx = encoder_cx.unsqueeze(0).repeat(inp.size(1), 1).unsqueeze(0)
        
        memory_bank, (hidden, c_n) = self.encoder(inp, inp_len, (encoder_hx, encoder_cx))
        memory_bank = memory_bank.transpose(0, 1) # [batch_size, seq_len, hidden_size]
        # hidden = hidden.transpose(0, 1) # [batch_size, 1, hidden_size]
        
        dec_i1 = torch.rand(memory_bank.shape[0], 1, memory_bank.shape[2])
        if torch.cuda.is_available(): dec_i1 = dec_i1.cuda()
    
        for i in range(self.process_block_iter):
            # memory_bank_pr, (hidden, c_n) = self.process_block(dec_i1)
            _, align_score, _ = self.attending(memory_bank, dec_i1, None, None)
            # dec_i1 = torch.bmm(align_score, memory_bank)
            dec_i1 = torch.einsum('bc,bch->bh', align_score.squeeze(1), memory_bank)
            dec_i1 = dec_i1.unsqueeze(1)
        outp = self.decoder(dec_i1)
        
        return outp
        
        