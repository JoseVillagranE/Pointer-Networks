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
                 process_block_iter, batch_size, dropout=0.0, C=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_process_block = process_block_iter
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.process_block = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)
        self.attending = Attention(hidden_dim, mask_bool=True, hidden_att_bool=False,
                                   C=None, device=self.device)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 1, bias=False))
        
        
        self.embedding = nn.Linear(2, embedding_dim, bias=False)
        self.dec_input = nn.Parameter(torch.FloatTensor(embedding_dim))
        
        self.sm = nn.Softmax()
        self.encoder = self.encoder.to(self.device)
        self.process_block = self.process_block.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.embedding = self.embedding.to(self.device)
            
        
        
        
    def forward(self, inp):
        
        '''
        input:
            inp: Muestras de Tour
            inp_len: cantidad de coordenas más una variable dummy
        output:
            outp: Baseline calculado por el critic.
        
        '''
        
        inp = self.embedding(inp) # [batch, seq_len, hidden_dim]
        # (encoder_hx, encoder_cx) = self.encoder.enc_init_state
        # encoder_hx = encoder_hx.unsqueeze(0).repeat(inp.size(1), 1).unsqueeze(0)       
        # encoder_cx = encoder_cx.unsqueeze(0).repeat(inp.size(1), 1).unsqueeze(0)
        
        enc_inp, (hidden, c_n) = self.encoder(inp, None)
        dec_input = self.dec_input.unsqueeze(0).repeat(enc_inp.shape[0],1).to(self.device) # [batch, emb_dim]
        for i in range(self.n_process_block):
            _, (hidden, c_n) = self.process_block(dec_input.unsqueeze(1), (hidden, c_n))
            g_l, align_score, _, _ = self.attending(enc_inp, hidden.squeeze(),
                                                   None, None)
        outp = self.decoder(g_l).squeeze()
        return outp
        
        