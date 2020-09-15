# coding=utf-8

import torch
from torch import nn
from torch.autograd import Variable

from layers.seq2seq.encoder import RNNEncoder
from layers.seq2seq.decoder import RNNDecoderBase
from layers.attention import Attention

import numpy as np
class PointerNetRNNDecoder(RNNDecoderBase):
    """ Pointer network RNN Decoder, process all the output together
    """
    def __init__(self, rnn_type, bidirectional, num_layers,
        input_size, hidden_size, dropout, batch_size, attn_type, C=None, is_cuda_available=False):
        super(PointerNetRNNDecoder, self).__init__(rnn_type, bidirectional, num_layers,
        input_size, hidden_size, dropout)
        #    self.attention = Attention("dot", hidden_size)
        
        if bidirectional:
            hidden_size *= 2
        self.attention = Attention(attn_type, hidden_size, batch_size, C=C, 
                                   is_cuda_available=is_cuda_available)
    
    def forward(self, tgt, memory_bank, hidden, inp, Teaching_Forcing=0):
        
        align_scores = []
        idxs = []
        logits = []
        memory_bank = memory_bank.transpose(0, 1)
        idx = torch.zeros((inp.size()))
        for i in range(tgt.shape[0]): # For each nodes
            
            if i == 0 or Teaching_Forcing > np.random.random():
                dec_i = tgt[i, :, :].unsqueeze(0)
            else:
                dec_i = inp[idx.data.squeeze(), [j for j in range(tgt.shape[1])],:].unsqueeze(0)            
            
            dec_outp, hidden = self.rnn(dec_i, hidden) # i=0 -> token
            dec_outp = dec_outp.transpose(0, 1)
            hidden_att, align_score, logit = self.attention(memory_bank, 
                                                            hidden[0][0].unsqueeze(0).transpose(0, 1), # dec_outp
                                                            training_type="Sup")
            
            idx = align_score.argmax(dim=2)
            align_scores.append(align_score)
            logits.append(logit)
            idxs.append(idx)
        align_scores = torch.stack(align_scores, dim=2).squeeze(-1)
        logits = torch.stack(logits, dim=2).squeeze(-1)
        # align_scores = align_scores.squeeze(-1)
        idxs = torch.stack(idxs, dim=1).squeeze(-1)
        return align_scores, logits, idxs
    
    
    
    
class PointerNetRNNDecoder_RL(RNNDecoderBase):
    """
    Decoder Network For RL training
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
        input_size, hidden_size, dropout, batch_size, attn_type, C=None, n_glimpses=1):
        super(PointerNetRNNDecoder_RL, self).__init__(rnn_type, bidirectional, num_layers,
        input_size, hidden_size, dropout)
        
        if bidirectional:
            hidden_size *= 2
        
        self.attention = Attention(attn_type, hidden_size, batch_size, C=C)
        self.glimpse = Attention(attn_type, hidden_size, batch_size, C=C)
        self.n_glimpses = n_glimpses
        self.sm = nn.Softmax()
        self.decoder = nn.LSTM(input_size, hidden_size)
        
    def forward(self, dec_inp_b, inp, memory_bank, hidden):
        
        """
        
        Se utiliza una elección Estocastica*
        
        input:
            dec_inp: entrada al decoder (dec_j. Donde dec_0 = <g>)
            inp: coordenadas de los nodos del TSP [len_tour, batch_size, emb_dim]
        return:
            align_scores: probs de salida
            rnn_outp: salida del dec
            idxs: indices-salidas del mecanismo de atención
        """
        selections = []#np.zeros((dec_inp_b.shape[1], inp.shape[0]))
        align_scores = []
        mask = None
        idxs = None
        dec_inp = dec_inp_b.unsqueeze(0) # [1, batch_size, hidden_size]. idx=0 -> token <g>
        for i in range(inp.shape[0]):
            dec_outp, next_hidden = self.decoder(dec_inp, hidden) #[seq_len=1, batch, hidden_size]
            if i == 0:
                memory_bank = memory_bank.transpose(0, 1) #[batch_size, seq_len, hidden_size]
            dec_outp = dec_outp.transpose(0, 1)# [batch_size, 1, hidden_size]
            for j in range(self.n_glimpses):
                _, align_score, _ = self.glimpse(memory_bank,
                                                 hidden[0][0].unsqueeze(0).transpose(0, 1),
                                                 None, None)
                if len(align_score.size())==1:
                    # dec_outp = torch.bmm(align_score, memory_bank)
                    dec_outp = torch.einsum('bc,bch->bh', align_score.squeeze(1), memory_bank)
                    dec_outp = dec_outp.unsqueeze(0)
                    # dec_outp = dec_outp.transpose(1, 2)
                    
                    if j == self.n_glimpses - 1:
                        dec_outp = dec_outp.transpose(1, 2)
                else:
                    # dec_outp = torch.bmm(align_score, memory_bank)
                    dec_outp = torch.einsum('bc,bch->bh', align_score.squeeze(1), memory_bank)
                    dec_outp = dec_outp.unsqueeze(0).transpose(0, 1)
            # align_score -> [batch, seq_len, 1]
            dec_inp = dec_inp.transpose(0, 1)
            _, align_score, mask = self.attention(memory_bank,
                                                  hidden[0][0].unsqueeze(0).transpose(0, 1), 
                                                  mask, idxs) # align_score -> [batch_size, #nodes]
            hidden = next_hidden
            align_score = align_score.squeeze()
            idxs = align_score.multinomial(num_samples=1).squeeze()
            # idxs = torch.argmax(align_score, dim=1)
            for old_idxs in selections:
                if old_idxs.eq(idxs).data.any():
                    idxs = align_score.multinomial(num_samples=1).squeeze()
                    break
            selections.append(idxs)
            align_scores.append(align_score)
            dec_inp = inp[idxs.data, [j for j in range(dec_inp_b.shape[0])],:].unsqueeze(0)
        
        if inp.shape[0]:
            selections = torch.stack(selections, dim=1)
            align_scores = torch.stack(align_scores, dim=1)
        else:
            selections = torch.stack(selections, dim=1).transpose(0, 1)
            align_scores = torch.stack(align_scores, dim=1).transpose(0, 1)

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
        encoder_input_size, rnn_hidden_size, dropout, batch_size, attn_type="Sup", C=None, 
        is_cuda_available = False):
        super().__init__()
        self.encoder = RNNEncoder(rnn_type, bidirectional,num_layers, encoder_input_size,
                                  rnn_hidden_size, dropout)
        self.attn_type = attn_type
        if attn_type == "Sup":
            self.decoder = PointerNetRNNDecoder(rnn_type, bidirectional,
                                    num_layers, encoder_input_size, rnn_hidden_size,
                                    dropout,batch_size, attn_type=attn_type, C=C,
                                    is_cuda_available=is_cuda_available)
        elif attn_type == "RL":
            self.decoder = PointerNetRNNDecoder_RL(rnn_type, bidirectional,
                                    num_layers, encoder_input_size, rnn_hidden_size, dropout, batch_size, attn_type="RL", C=C)
         
      
    def forward(self, inp, inp_len, outp, outp_len, Teaching_Forcing=0):
        
        inp = inp.transpose(0, 1) # [batch, seq_len, emb_size]
        
        if self.attn_type == "Sup":
            outp = outp.transpose(0, 1)# [seq_len, batch, emb_size]
            memory_bank, hidden = self.encoder(inp, inp_len)
            align_score, logits, idxs = self.decoder(outp, memory_bank, hidden, inp, Teaching_Forcing=Teaching_Forcing)
            return align_score, logits, idxs
        elif self.attn_type == "RL":
            
            (encoder_hx, encoder_cx) = self.encoder.enc_init_state
            encoder_hx = encoder_hx.unsqueeze(0).repeat(inp.size(1), 1).unsqueeze(0)       
            encoder_cx = encoder_cx.unsqueeze(0).repeat(inp.size(1), 1).unsqueeze(0)
            # memory_bank -> [seq_len, batch, hidden_size]
            # hidden_0 -> [1, batch, hidden_size]
            
            memory_bank, hidden_final = self.encoder(inp, inp_len, (encoder_hx, encoder_cx))
            align_score, idxs, dec_memory_bank  = self.decoder(outp, inp, memory_bank, hidden_final)
            return align_score, memory_bank, dec_memory_bank, idxs


def sequence_mask(lengths, maxlen, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    a = torch.ones((len(lengths), maxlen))
    if torch.cuda.is_available():
        a = a.cuda()
        
    mask = ~(a.cumsum(dim=1).t() > lengths.float()).t()
    mask.type(dtype)
    return mask


class PointerNetLoss(nn.Module):
    """ Loss function for pointer network
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-15
    
    def forward(self, target, logits, lengths):
        """
        Args:
        target : label data (bz, tgt_max_len)
        logits : predicts (bz, tgt_max_len, src_max_len)
        lengths : length of label data (bz)
        """
        
        logits = logits.clamp(min=self.eps)
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