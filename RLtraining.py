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
from torch.nn.utils import clip_grad_norm_

from layers.seq2seq.encoder import RNNEncoder
from layers.seq2seq.decoder import RNNDecoderBase
from layers.attention import Attention

from CriticNetwork import CriticNetwork
from torch.utils.data import DataLoader
from pointer_network import PointerNet, PointerNetLoss
from torch import optim

from time import time
from TSP import PreProcessOutput
from TSPDataset import TSPDataset, Generator
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
    for name, param in module.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0.0)
        else:
            nn.init.uniform_(param, a=a, b=b)



def Reward(sample_solution, is_cuda_available=False):
    '''
    input:
        sample_solution: tensor que contiene la solución de un tour ([batch, seq_len, 2])
    
    output: 
        tour_length: tensor que contiene el largo de cada tour 
    '''
    batch_size, n_nodes, _ = sample_solution.size()
    tour_length = torch.zeros([batch_size])
    if is_cuda_available:
        tour_length = tour_length.cuda()
    
    for i in range(n_nodes-1):
        tour_length += torch.norm(sample_solution[:, i] - sample_solution[:, i+1], p=2, dim=1)
    
    # final trip
    tour_length += torch.norm(sample_solution[:, n_nodes-1] - sample_solution[:, 0], p=2, dim=1)
    return tour_length


def tensor_sort(input, idxs, dim=1):
    
    """
    input:
        input: tensor
        idxs: indices por el cual el tensor será ordenado
    outp:
        outp: tensor ordenado
    """
    if dim==1:
        outp = torch.gather(input, dim=dim, index=idxs[:, :, None].repeat(1,1,2))
    else:
        outp = torch.gather(input, dim=dim, index=idxs[:, :, None])
    return outp
            


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

class NeuronalOptm:
    
    def __init__(self, input_lenght, rnn_type, bidirectional, num_layers, rnn_hidden_size, 
                 embedding_dim, hidden_dim_critic, process_block_iter,
                 inp_len_seq, lr, C=None, batch_size=10, training_type="RL", actor_decay_rate=0.96,
                 critic_decay_rate=0.99, step_size=50):
        
        super().__init__()
        self.model = PointerNet(rnn_type, bidirectional, num_layers, embedding_dim,
                       rnn_hidden_size, batch_size=batch_size,
                       training_type=training_type, C=C)
        
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
        self.dec_0.data.uniform_(0, 1)
        # self.embedding = nn.Parameter(embedding)
        
        # self.dec_0.data.uniform_(-(1. / math.sqrt(self.embedding_dim)), 1. / math.sqrt(self.embedding_dim))
        # self.embedding.data.uniform_(-(1. / math.sqrt(self.embedding_dim)), 1. / math.sqrt(self.embedding_dim))
        
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
        
        
        if self.is_cuda_available:
            self.model = self.model.cuda()
            self.critic = self.critic.cuda()
            self.dec_0 = self.dec_0.cuda()
            
        
    def step(self, batch_inp, clip_norm=1.0):
        
        align_score, memory_bank, dec_memory_bank, idxs = self.model(batch_inp)
        baseline = self.critic(batch_inp)
        sample_solution = tensor_sort(batch_inp, idxs, dim=1).squeeze() # [batch, seq_len, 2]
        sample_probs = tensor_sort(align_score, idxs, dim=2).squeeze() # [batch, seq_len]
        
        tour_length = Reward(sample_solution, self.is_cuda_available) #[batch]
        
        log_probs = torch.log(sample_probs).sum(dim=1)
        
        adv = tour_length.detach() - baseline.detach()
        actor_loss = (adv*log_probs).mean()
        
        self.optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.model.parameters(), clip_norm)
        self.optimizer.step()
        
        self.optim_critic.zero_grad()
        critic_loss = self.critic_loss(baseline, tour_length.detach())
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), clip_norm)
        self.optim_critic.step()
        
        self.actor_lr_sch.step()
        self.critic_lr_sch.step()
        
        actor_loss_item = actor_loss.detach().item()
        critic_loss_item = critic_loss.detach().item()
        tour_length_mean = tour_length.detach().mean()
        return actor_loss_item, critic_loss_item, tour_length_mean
    
    def training(self, train_ds, eval_ds, attention_size=128, beam_width=2,
                 lr=1e-3, clip_norm=1.0, weight_decay=0.1, nepoch = 50, 
                 save_model_file="RLPointerModel.pt", freqEval=5):
        
        
        t0 = time()
  
        train_dl = DataLoader(train_ds, batch_size=self.batch_size)
        # eval_dl = DataLoader(eval_ds, batch_size=self.batch_size)

        
        list_of_actor_loss = []
        list_of_critic_loss = []
        list_of_tour_length_mean = []
        for epoch in range(nepoch):
            
            self.model = self.model.train()
            self.critic = self.critic.train()
            actor_total_loss = 0.
            critic_total_loss = 0.
            tour_length_total = 0.
            batch_cnt = 0.
            steps = 0
            for step, b_inp in enumerate(train_dl):
                b_inp = Variable(b_inp)
                
                if self.is_cuda_available:
                    b_inp = b_inp.cuda()
                actor_loss, critic_loss, tour_length_mean = self.step(b_inp,
                                                                      clip_norm=clip_norm)
                actor_total_loss += actor_loss
                critic_total_loss += critic_loss
                tour_length_total += tour_length_mean
                batch_cnt += 1
                if (step+1)%10 == 0:
                    print(f"Step: {step} ||", end=' ')
                    print(f"Actor Loss:  {actor_total_loss / (step+1):.6f} ||", end=' ') 
                    print(f"Critic Loss: {critic_total_loss/(step+1):.3f} ||", end=' ')
                    print(f"Tour Length: {tour_length_total/(step+1):.2f}")
            list_of_actor_loss.append(actor_total_loss/batch_cnt)
            list_of_critic_loss.append(critic_total_loss/batch_cnt)
            list_of_tour_length_mean.append(tour_length_total/batch_cnt)
            # eval_model(self.model, self.embedding, eval_ds, self.is_cuda_available, self.batch_size)
            
        
        torch.save(self.model.state_dict(), save_model_file)
        t1 = time()
        t = t1 - t0
        hours, rem = divmod(t, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Training of Pointer Networks takes: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),
                                                                                  int(minutes),
                                                                                  seconds))
        
        return list_of_actor_loss, list_of_critic_loss, list_of_tour_length_mean
    
    
    def eval_model(self, eval_ds, batch_size=1, n_plt_tours=9, n_cols=3, beam_serch=None):
        
        
        device = 'cuda:0' if torch.cuda.is_available else 'cpu'
        self.model = self.model.eval().to(device)
        countAcc = 0
        eval_dl = DataLoader(eval_ds, num_workers=0, batch_size=batch_size, shuffle=True)
        for b_eval_inp, b_eval_inp_len, b_eval_outp_in, b_eval_outp_out, b_eval_outp_len in eval_dl:
        
            b_eval_inp = Variable(b_eval_inp).to(device)
            b_eval_outp_in = Variable(b_eval_outp_in).to(device)
            b_eval_outp_out = Variable(b_eval_outp_out).to(device)
        
        
            align_score, _, _, idxs = self.model(b_eval_inp)
            align_score = align_score.cpu().detach().numpy()
            idxs = idxs.cpu().detach().numpy()
            labels = b_eval_outp_out.cpu().detach().numpy().squeeze()
            if functools.reduce(lambda i, j: i and j, map(lambda m, k: m==k, idxs, labels), True):
                countAcc += 1
    
        Acc = countAcc/eval_ds.__len__()
        print("The Accuracy of the model is: {}".format(Acc))
        
        
    
    def plot_one_tour(self, example):
    
        self.model.eval()
        inp, inp_len, outp_in, outp_out, outp_len = example
        inp_t = Variable(torch.from_numpy(np.array([inp])))
        inp_len = torch.from_numpy(inp_len)
        outp_in = Variable(torch.from_numpy(np.array([outp_in])))
        outp_out = Variable(torch.from_numpy(outp_out))
        
        embedding = self.embedding.cpu()
        embedded_inputs = embedding(inp_t).cpu()
        dec_0_i = self.dec_0.unsqueeze(0)
        # embedded_inputs = []
        # # result is [batch_size, 1, seq_len, inp_dim] 
        # ips = inp_t.unsqueeze(1)
        
        # for i in range(self.seq_len):
        #     # [batch_size x 1 x input_dim] * [batch_size x input_dim x embedding_dim]
        #     # result is [batch_size, embedding_dim]
        #     embedded_inputs.append(torch.bmm(
        #         ips[:, :, i, :].float(),
        #         embedding_i).squeeze(1))
            
        
        # # Result is [ batch_size x sourceL x embedding_dim]
        # embedded_inputs = Variable(torch.cat(embedded_inputs).view(1, self.seq_len,
        #                                             self.embedding_dim), requires_grad=False)
        
        
        if torch.cuda.is_available():
            embedded_inputs = embedded_inputs.cuda()
        
        align_score, _, _, idxs = self.model(embedded_inputs, inp_len, dec_0_i, outp_len)
        # align_score = align_score[0].detach().numpy()
        # idxs = np.argmax(align_score.cpu(), axis=1)
        # idxs = PreProcessOutput(idxs)
        
        # inp = inp[1:, :]
        
        idxs = idxs.cpu().numpy()
        
        plt.figure(figsize=(10,10))
        plt.plot(inp[:,0], inp[:,1], 'o')
        for i in range(idxs.shape[0]-1):
            plt.plot(inp[[idxs[i], idxs[i+1]],0], inp[[idxs[i], idxs[i+1]], 1], 'k-')

if __name__ == "__main__":
    
    train_filename="./CH_TSP_data/tsp5.txt" 
    val_filename = "./CH_TSP_data/tsp5_test.txt"

    seq_len = 5
    num_layers = 1 # Se procesa con sola una celula por coordenada. 
    input_lenght = 2 
    rnn_hidden_size = 128
    rnn_type = 'LSTM'
    bidirectional = False
    hidden_dim_critic = rnn_hidden_size
    process_block_iter = 1
    inp_len_seq = seq_len
    lr = 1e-2
    C = 10 # Logit clipping
    batch_size = 512
    n_epoch=1
    steps = 1
    embedding_dim = 128 #d-dimensional embedding dim
    embedding_dim_critic = embedding_dim
    f_city_fixed=False
    beam_search = None
    
    save_model_file="RLPointerModel_TSP5.pt"
    
    train_ds = TSPDataset(train_filename, f_city_fixed=f_city_fixed, lineCountLimit=1000)
    eval_ds = TSPDataset(val_filename, f_city_fixed=f_city_fixed, lineCountLimit=1000)
    
    train_ds = Generator(batch_size*steps, seq_len)
    
    print("Train data size: {}".format(len(train_ds)))
    print("Eval data size: {}".format(len(eval_ds)))
    
    trainer = NeuronalOptm(input_lenght, rnn_type, bidirectional, num_layers, rnn_hidden_size, 
                           embedding_dim, hidden_dim_critic, process_block_iter, inp_len_seq, lr, 
                           C=C, batch_size=batch_size)
    
    Actor_Training_Loss, Critic_Training_Loss, Tour_training_mean = trainer.training(train_ds, eval_ds,
                                                                                        save_model_file=save_model_file,
                                                                                        nepoch=n_epoch)
    
    trainer.eval_model(eval_ds)
        
        