# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:40:58 2020

@author: joser
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.utils import clip_grad_norm

from layers.seq2seq.encoder import RNNEncoder
from layers.seq2seq.decoder import RNNDecoderBase
from layers.attention import Attention

from CriticNetwork import CriticNetwork
from torch.utils.data import DataLoader
from pointer_network import PointerNet, PointerNetLoss
from torch import optim

from time import time
from TSP import PreProcessOutput
from TSPDataset import TSPDataset
import functools

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


layers_of_interest = ["Linear", "Conv1d"]

def weights_init(module, a=-0.08, b=0.08):
    
    """
    input:
        Module -> Neural Network Module
        a -> int. LowerBound
        b -> int. UpperBound
    """
    classname = module.__class__.__name__
    if classname in layers_of_interest:
        nn.init.uniform_(module.weight, a=a, b=b)
    elif classname.find("LSTM") != -1:
        for param in module.parameters():
            nn.init.uniform_(param.data)



def Reward(sample_solution, is_cuda_available=False):
    '''
    input:
        sample_solution: tensor que contiene la solución de un tour
    
    output: 
        tour_length: tensor que contiene el largo de cada tour 
    '''
    
    batch_size = sample_solution.shape[0]
    number_of_nodes = sample_solution.shape[1]
    tour_length = Variable(torch.zeros([batch_size]))
    
    if is_cuda_available:
        tour_length = tour_length.cuda()
    
    for i in range(number_of_nodes-1):
        tour_length += torch.norm(sample_solution[:, i, :] - sample_solution[:, i+1, :], dim=1)
        # tour_length += tour_length_i.cpu().numpy()
    
    # final trip
    tour_length += torch.norm(sample_solution[:, number_of_nodes-1, :] - sample_solution[:, 0, :], dim=1)
    # tour_length += tour_length_i.cpu().numpy()
    
    # tour_length = torch.from_numpy(tour_length)
        
    return tour_length


def tensor_sort(input, idxs, axis=1):
    
    """
    input:
        input: tensor
        idxs: indices por el cual el tensor será ordenado
    outp:
        outp: tensor ordenado
    """
    
    d1, d2, d3 = input.size()
    if axis==1:
        outp = input[torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(), idxs.flatten(), :].view(d1, d2, d3)
    else:
        outp = torch.gather(input, 2, idxs.unsqueeze(2))
    return outp
            

def PreProcces_outp_in(batch_outp_in):
    
    '''
    input:
        batch_outp_in: Recibe un batch que contiene la información de etiquetas
    output:
        batch_outp_in_process: Retorna las misma información, pero procesada convenientemente
    '''
    
    START = [0, 0]
    pass

def PreProcessOutput_batch(batch):
    
    """
    input:
        outp(numpy): tour o label
    output:
        outp(numpy): tour o label editado (-1)
    """
    outp = np.zeros((batch.shape[0], batch.shape[1] - 1))
    for i in range(batch.shape[0]):
        outp_ = np.array([batch[i, j] - 1 for j in range(batch.shape[1]) if batch[i, j] != 0])# Si la predicción arroja más de un 0, esto me arrebata varios idxs.
        outp[i, :] = outp_
    return outp

def eval_model(model, embedding, eval_ds, cudaAvailable, batchSize=1):
    model.eval()
    if cudaAvailable:
         use_cuda = True
         torch.cuda.device(0)
         model = model.cuda()
    else:
        use_cuda = False
        
    countAcc = 0
    eval_dl = DataLoader(eval_ds, num_workers=0, batch_size=batchSize)
    for b_eval_inp, b_eval_inp_len, b_eval_outp_in, b_eval_outp_out, b_eval_outp_len in eval_dl:
        
        b_eval_inp = Variable(b_eval_inp)
        b_eval_outp_in = Variable(b_eval_outp_in)
        b_eval_outp_out = Variable(b_eval_outp_out)
        
        if use_cuda:
            b_eval_inp = b_eval_inp.cuda()
            b_eval_inp_len = b_eval_inp_len.cuda()
            b_eval_outp_in = b_eval_outp_in.cuda()
            b_eval_outp_out = b_eval_outp_out.cuda()
            b_eval_outp_len = b_eval_outp_len.cuda()
        
        
        align_score, _, _ = model(b_eval_inp, b_eval_inp_len, embedding, b_eval_outp_len)
        align_score = align_score.cpu().detach().numpy()
        idxs = np.argmax(align_score, axis=2)
        b_eval_outp_out = b_eval_outp_out.cpu().detach().numpy()
        b_eval_outp_out = b_eval_outp_out.squeeze()
        # idxs = PreProcessOutput_batch(idxs)
        # labels = PreProcessOutput(b_eval_outp_out)
        # labels = PreProcessOutput_batch(b_eval_outp_out)
        labels = b_eval_outp_out
        print("labels: {}".format(labels))
        print("idxs: {}".format(idxs))
        if functools.reduce(lambda i, j: i and j, map(lambda m, k: m==k, idxs, labels), True):
            countAcc += 1
    
    Acc = countAcc/eval_ds.__len__()
    print("The Accuracy of the model is: {}".format(Acc))


class NeuronalOptm:
    
    def __init__(self, input_lenght, rnn_type, bidirectional, num_layers, rnn_hidden_size, 
                 embedding_dim, hidden_dim_critic, process_block_iter,
                 inp_len_seq, lr, C=None, batch_size=10, attn_type="RL", actor_decay_rate=0.96,
                 critic_decay_rate=0.99, step_size=5000):
        
        super().__init__()
        self.model = PointerNet(rnn_type, bidirectional, num_layers, embedding_dim, rnn_hidden_size, 0, batch_size, attn_type=attn_type, C=C)
        self.model.apply(weights_init)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.seq_len = inp_len_seq
        
        # enconder_input_size: dimension de cada coordenada(eg: 2)
        # rnn_hidden_size: dimensión del vector de embedding hidden
        
        # reemplazo de los labels de supervised-learning
        
        dec_0 = torch.FloatTensor(embedding_dim)
        embedding = torch.FloatTensor(input_lenght, self.embedding_dim)
        
        self.dec_0 = nn.Parameter(dec_0)
        self.embedding = nn.Parameter(embedding)
        
        self.dec_0.data.uniform_(-(1. / math.sqrt(self.embedding_dim)), 1. / math.sqrt(self.embedding_dim))
        self.embedding.data.uniform_(-(1. / math.sqrt(self.embedding_dim)), 1. / math.sqrt(self.embedding_dim))
        
        # result is [batch_size x inp_seq_len x 2]
        # self.dec_0 = self.dec_0.unsqueeze(0).repeat(batch_size, 1)
        
        
        self.is_cuda_available = torch.cuda.is_available()
        
        self.critic = CriticNetwork(rnn_type, num_layers, bidirectional, embedding_dim,
                                    hidden_dim_critic, process_block_iter, batch_size, C=C, is_cuda_available=self.is_cuda_available)
        
        self.critic.apply(weights_init)
        
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_loss = torch.nn.MSELoss()
        
        self.actor_lr_sch = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size,
                                                      gamma = actor_decay_rate)
        self.critic_lr_sch = optim.lr_scheduler.StepLR(self.optim_critic,
                                                              step_size=step_size, 
                                                              gamma=critic_decay_rate)
        
        self.dec_0 = self.dec_0.unsqueeze(0).repeat(self.batch_size, 1)
        self.embedding = self.embedding.unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        if self.is_cuda_available:
            self.model = self.model.cuda()
            self.embedding = self.embedding.cuda()
            self.dec_0 = self.dec_0.cuda()
            
        
    def step(self, batch_inp, batch_inp_len, batch_outp_out, batch_outp_len, clip_norm=10.0):
        
        # De momento se obtiene el largo del batch de etiquetas desde el dataset. Esta mal, pero es para no
        # perder tiempo
        embedded_inputs = []
        # result is [batch_size, 1, seq_len, inp_dim] 
        ips = batch_inp.unsqueeze(1)
        
        for i in range(self.seq_len):
            # [batch_size x 1 x input_dim] * [batch_size x input_dim x embedding_dim]
            # result is [batch_size, embedding_dim]
            embedded_inputs.append(torch.bmm(
                ips[:, :, i, :].float(),
                self.embedding).squeeze(1))
            
        
        # Result is [sourceL x batch_size x embedding_dim]
        embedded_inputs = Variable(torch.cat(embedded_inputs).view(self.batch_size, self.seq_len,
                                                    self.embedding_dim), requires_grad=False)   
        # Output of actor net
        align_score, memory_bank, dec_memory_bank, idxs = self.model(embedded_inputs, batch_inp_len, self.dec_0, batch_outp_len)
        
        # cnt=0
        # for param in self.model.parameters():
        #     if cnt==0:
        #         print(param.data)
        #         break
        
        #Output of critic net
        # memory_bank = memory_bank.transpose(0, 1).detach()
        baseline = self.critic(embedded_inputs, batch_inp_len)
        baseline = baseline.squeeze()
        sample_solution = tensor_sort(batch_inp, idxs)
        sample_probs = tensor_sort(align_score, idxs, axis=2)
        # print("sample_solution: {}".format(sample_solution))
        tour_length = Reward(sample_solution, self.is_cuda_available)
        
        log_probs = torch.log(sample_probs.sum(dim=1))
        nll = -1*log_probs     
        
        # En caso que hayan nan's
        nll[(nll != nll).detach()] = 0.
        # no forzar el gradiente a grandes números
        log_probs[(log_probs < -1000).detach()] = 0.
        
        actor_loss = abs(tour_length - baseline.detach())*log_probs
        
        actor_loss = actor_loss.mean()
        
        self.optimizer.zero_grad()
        self.optim_critic.zero_grad()
        actor_loss.backward(retain_graph=True)
        actor_loss_item = actor_loss.item()
        clip_grad_norm(self.model.parameters(), clip_norm)
        
        self.optimizer.step()
        self.actor_lr_sch.step()
        
        critic_loss = self.critic_loss(baseline, tour_length.detach())
        critic_loss.backward()
        critic_loss_item = critic_loss.item()
        clip_grad_norm(self.critic.parameters(), clip_norm)
        
        
        self.optim_critic.step()
        self.critic_lr_sch.step()
        
        tour_length_mean = tour_length.mean()
        return actor_loss_item, critic_loss_item, tour_length_mean
    
    def training(self, train_ds, eval_ds, attention_size=128, beam_width=2,
                 lr=1e-3, clip_norm=5.0, weight_decay=0.1, nepoch = 50, 
                 save_model_file="RLPointerModel.pt", freqEval=5):
        
        
        t0 = time()
  
        train_dl = DataLoader(train_ds, num_workers=0, batch_size=self.batch_size)
        # eval_dl = DataLoader(eval_ds, num_workers=0, batch_size=self.batch_size)

        
        list_of_actor_loss = []
        list_of_critic_loss = []
        list_of_tour_length_mean = []
        steps = 0
        for epoch in range(nepoch):
            
            self.model = self.model.train()
            actor_total_loss = 0.
            critic_total_loss = 0.
            tour_length_total = 0.
            batch_cnt = 0.
            for b_inp, b_inp_len, b_outp_in, b_outp_out, b_outp_len in train_dl:
                b_inp = Variable(b_inp)
                b_outp_out = Variable(b_outp_out)
                b_inp_len = Variable(b_inp_len)
                b_outp_len = Variable(b_outp_len)
                if self.is_cuda_available:
                    b_inp = b_inp.cuda()
                    b_inp_len = b_inp_len.cuda()
                    b_outp_in = b_outp_in.cuda()
                    b_outp_out = b_outp_out.cuda()
                    b_outp_len = b_outp_len.cuda()
                
                actor_loss, critic_loss, tour_length_mean = self.step(b_inp, b_inp_len, b_outp_out, b_outp_len, clip_norm=clip_norm)
                steps += 1
                actor_total_loss += actor_loss
                critic_total_loss += critic_loss
                tour_length_total += tour_length_mean
                batch_cnt += 1
            print("Epoch: {0} || N_steps: {1} || Actor Loss:  {2:.6f} || Critic Loss: {3:.3f} || Tour Length: {4:.2f}".format(epoch, steps, actor_total_loss / batch_cnt, critic_total_loss/batch_cnt, tour_length_mean))
            list_of_actor_loss.append(actor_total_loss/batch_cnt)
            list_of_critic_loss.append(critic_total_loss/batch_cnt)
            list_of_tour_length_mean.append(tour_length_total/batch_cnt)
            # eval_model(self.model, self.embedding, eval_ds, self.is_cuda_available, self.batch_size)
            
        
        torch.save(self.model.state_dict(), save_model_file)
        t1 = time()
        t = t1 - t0
        hours, rem = divmod(t, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Training of Pointer Networks takes: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        
        fig_1 = plt.figure(1)
        ax1 = fig_1.add_subplot(1, 1, 1)
        ax1.plot(range(len(list_of_tour_length_mean)), list_of_tour_length_mean)
        fig_2 = plt.figure(2)
        ax2 = fig_2.add_subplot(1, 1, 1)
        ax2.plot(range(len(list_of_critic_loss)), list_of_critic_loss)
        
        return list_of_actor_loss, list_of_critic_loss, list_of_tour_length_mean

if __name__ == "__main__":
    
    train_filename="./CH_TSP_data/tsp_all_len20.txt" 
    val_filename = "./CH_TSP_data/tsp_20_test.txt"

    seq_len = 20
    num_layers = 1 # Se procesa con sola una celula por coordenada. 
    input_lenght = 2 
    rnn_hidden_size = 128
    rnn_type = 'LSTM'
    bidirectional = False
    hidden_dim_critic = rnn_hidden_size
    process_block_iter = 3
    inp_len_seq = seq_len
    lr = 1e-3
    C = 1
    batch_size = 128
    n_epoch = 50
    embedding_dim = 128 #d-dimensional embedding dim
    # encoder_input_size = embedding_dim
    embedding_dim_critic = embedding_dim
    
    save_model_file="RLPointerModel_TSP20.pt"
    
    train_ds = TSPDataset(train_filename, seq_len, lineCountLimit=1000)
    eval_ds = TSPDataset(val_filename, seq_len, lineCountLimit=100)
    
    print("Train data size: {}".format(len(train_ds)))
    print("Eval data size: {}".format(len(eval_ds)))
    
    trainer = NeuronalOptm(input_lenght, rnn_type, bidirectional, num_layers, rnn_hidden_size, 
                           embedding_dim, hidden_dim_critic, process_block_iter, inp_len_seq, lr, 
                           C=C, batch_size=batch_size)
    
    Actor_Training_Loss, Critic_Training_Loss, Tour_training_mean = trainer.training(train_ds, eval_ds,
                                                                                     save_model_file=save_model_file,
                                                                                     nepoch=n_epoch)
        
        
        
        