# coding=utf-8

import torch
from torch import nn
from torch.autograd import Variable

from layers.seq2seq.encoder import RNNEncoder
from layers.seq2seq.decoder import RNNDecoderBase
from layers.attention import Attention, sequence_mask

import numpy as np
class PointerNetRNNDecoder(RNNDecoderBase):
    """ Pointer network RNN Decoder, process all the output together
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
        input_size, hidden_size, dropout, batch_size, attn_type, C=None):
        super(PointerNetRNNDecoder, self).__init__(rnn_type, bidirectional, num_layers,
        input_size, hidden_size, dropout)
        #    self.attention = Attention("dot", hidden_size)
        self.attention = Attention(attn_type, hidden_size, batch_size, C=C)
    
    def forward(self, tgt, memory_bank, hidden, memory_lengths=None):
        # RNN
        rnn_output, hidden_final = self.rnn(tgt, hidden)
        # Attention
        memory_bank = memory_bank.transpose(0, 1)
        rnn_output = rnn_output.transpose(0, 1)
        attn_h, align_score = self.attention(memory_bank, rnn_output, memory_lengths)
        
        return align_score, rnn_output
    
class PointerNetRNNDecoder_RL(RNNDecoderBase):
    """
    Decoder Network For RL training
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
        input_size, hidden_size, dropout, batch_size, attn_type, C=None, n_glimpses=1):
        super(PointerNetRNNDecoder_RL, self).__init__(rnn_type, bidirectional, num_layers,
        input_size, hidden_size, dropout)
        
        self.attention = Attention(attn_type, hidden_size, batch_size, C=C)
        self.glimpse = Attention(attn_type, hidden_size, batch_size, C=C)
        self.n_glimpses = n_glimpses
        self.sm = nn.Softmax()
        
    def forward(self, dec_inp_b, inp, memory_bank, hidden, memory_lengths=None):
        
        """
        
        Se utiliza una elección Estocastica*
        
        input:
            dec_inp: entrada al decoder (dec_j. Donde dec_0 = <g>)
            inp: coordenadas de los nodos del TSP [len_tour, batch_size, 2]
        return:
            align_scores: probs de salida
            rnn_outp: salida del dec
            idxs: indices-salidas del mecanismo de atención
        """
        selections = []#np.zeros((dec_inp_b.shape[1], inp.shape[0]))
        align_scores = []
        dec_inp = dec_inp_b[0, :, :].unsqueeze(0) # [1, batch_size, 2]. idx=0 -> token
        for i in range(inp.shape[0]):
            rnn_outp, hidden_final = self.rnn(dec_inp, hidden)
            hidden = hidden_final
            if i == 0:
                memory_bank = memory_bank.transpose(0, 1) #[batch_size, emb_size, hidden_size]
            rnn_outp = rnn_outp.transpose(0, 1)# [batch_size, 1, hidden_size]
            for _ in range(self.n_glimpses):
                attn_h, align_score = self.glimpse(memory_bank, rnn_outp)
                attn_h = attn_h.transpose(1, 2)
                rnn_outp = torch.bmm(self.sm(align_score), attn_h)
            

            attn_h, align_score = self.attention(memory_bank, rnn_outp, memory_lengths) # align_score -> [batch_size, #nodes]
            align_score = self.sm(align_score.squeeze())
            idxs = align_score.multinomial(num_samples=1).squeeze()
            for old_idxs in selections:
                if old_idxs.eq(idxs).data.any():
                    idxs = align_score.multinomial(num_samples=1).squeeze()
                    break
            selections.append(idxs)
            align_scores.append(align_score)
            dec_inp = inp[idxs.data, [j for j in range(dec_inp_b.shape[1])],:].unsqueeze(0)

        selections = torch.stack(selections).transpose(0, 1)
        align_scores = torch.stack(align_scores).transpose(0, 1)
        
        return align_scores, selections, hidden
        
            
class PointerNet(nn.Module):
    """ Pointer network
    Args:
    rnn_type (str) : rnn cell type
    bidirectional : whether rnn is bidirectional
    num_layers : number of layers of stacked rnn
    encoder_input_size : input size of encoder
    rnn_hidden_size : rnn hidden dimension size
    dropout : dropout rate
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
        encoder_input_size, rnn_hidden_size, dropout, batch_size, attn_type="dot", C=None):
        super().__init__()
        self.encoder = RNNEncoder(rnn_type, bidirectional,
        num_layers, encoder_input_size, rnn_hidden_size, dropout)
        if attn_type in ['dot', 'general']:
            self.decoder = PointerNetRNNDecoder(rnn_type, bidirectional,
                                    num_layers, encoder_input_size, rnn_hidden_size, dropout,batch_size, attn_type=attn_type, C=C)
        else:
            self.decoder = PointerNetRNNDecoder_RL(rnn_type, bidirectional,
                                    num_layers, encoder_input_size, rnn_hidden_size, dropout, batch_size, attn_type=attn_type, C=C)
                
      
    def forward(self, inp, inp_len, outp, outp_len):
        inp = inp.transpose(0, 1)
        outp = outp.transpose(0, 1)
        memory_bank, hidden_final = self.encoder(inp, inp_len)
        align_score, idxs, dec_memory_bank  = self.decoder(outp, inp, memory_bank, hidden_final, inp_len)
        return align_score, memory_bank, dec_memory_bank, idxs

class PointerNetLoss(nn.Module):
    """ Loss function for pointer network
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, target, logits, lengths):
        """
        Args:
        target : label data (bz, tgt_max_len)
        logits : predicts (bz, tgt_max_len, src_max_len)
        lengths : length of label data (bz)
        """
        
        _, tgt_max_len = target.size()
        logits_flat = logits.view(-1, logits.size(-1))
        log_logits_flat = torch.log(logits_flat)
        target_flat = target.view(-1, 1).long()
        losses_flat = -torch.gather(log_logits_flat, dim=1, index = target_flat)
        losses = losses_flat.view(*target.size())
        mask = sequence_mask(lengths, tgt_max_len)
        mask = Variable(mask)
        losses = losses * mask.float()
        loss = losses.sum() / lengths.float().sum()
        return loss