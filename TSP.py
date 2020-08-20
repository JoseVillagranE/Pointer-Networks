# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:52:01 2020

@author: joser
"""

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import sys
import matplotlib.pyplot as plt

import functools
from time import time
import random
from TSPDataset import TSPDataset
from utils import compute_len_tour, count_valid_tours
from pointer_network import PointerNet, PointerNetLoss
import warnings

from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

'''
TODO:
    - implementar un formato de evaluación supervisado
    - Mecanismo de predicción sin label. (test time)
    
    - Refactorizar todoooo !!
'''

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
            
    

def PreProcessOutput(outp):
    
    outp = [outp[i] - 1 for i in range(outp.shape[0]) if outp[i] != 0]
    return outp


def eval_model(model, eval_ds, cudaAvailable, batchSize=1, n_plt_tours=0, n_cols=1):
    model.eval()
    if cudaAvailable:
         use_cuda = True
         torch.cuda.device(0)
         model = model.cuda()
    else:
        use_cuda = False
        
    countAcc = 0
    n_invalid_tours = 0
    len_tours = 0
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
        
        align_score, idxs = model(b_eval_inp, b_eval_inp_len, b_eval_outp_in, b_eval_outp_len, Teaching_Forcing=0)
        align_score = align_score.cpu().detach().numpy()
        # idxs = np.argmax(align_score, axis=2)
        idxs = idxs.cpu().numpy()
        b_eval_outp_out = b_eval_outp_out.cpu().detach().numpy()
        b_eval_outp_out = b_eval_outp_out.squeeze()
        idxs = PreProcessOutput(idxs.squeeze())
        labels = PreProcessOutput(b_eval_outp_out)
        
        # print(idxs)
        # print(labels)
        
        # Sirve solo si batch_size = 1
        if functools.reduce(lambda i, j: i and j, map(lambda m, k: m==k, idxs, labels), True):
            # Evaluación estricta
            countAcc += 1
            
        if len(idxs) != len(set(idxs)):
            n_invalid_tours += 1
        else:
            len_tour = compute_len_tour(b_eval_inp.cpu().detach().numpy(), idxs)
            len_tours += len_tour
    
    Acc = countAcc/eval_ds.__len__() # cuidado al momento de evaluar en batch
    len_tour_mean = len_tours/eval_ds.__len__()
    print("The Accuracy of the model is: {}".format(Acc))
    print("Number of Invalid Tours: {}".format(n_invalid_tours))
    print("Total Number of Tours: {}".format(eval_ds.__len__()))
    print("Avg Tour Length: {:.3f}".format(len_tour_mean))
    n_rows = n_plt_tours // n_cols
    n_rows += n_plt_tours % n_cols
    pos = range(1, n_plt_tours+1)
    
    i_tour_alrea_sel = []
    # fig, ax = plt.subplots(n_plt_tours, figsize=(10,10))
    fig = plt.figure(figsize=(10,10))
    for i in range(n_plt_tours):
        
        Run = True
        while Run:
            i_tour = np.random.randint(0, len(eval_ds))
            if not i_tour in i_tour_alrea_sel:
                Run = False
                i_tour_alrea_sel.append(i_tour)
        example = eval_ds.__getitem__(i_tour)
        # ax_ = ax[i]
        ax = fig.add_subplot(n_rows, n_cols, pos[i])
        plot_one_tour(model, example, ax, cudaAvailable)
    
    plt.show()
    

def plot_one_tour(model, example, ax=None, cudaAvailable=False):
    
    if cudaAvailable:
        model = model.cuda()
    model.eval()
    inp, inp_len, outp_in, outp_out, outp_len = example
    inp_t = Variable(torch.from_numpy(np.array([inp])))
    inp_len = torch.from_numpy(inp_len)
    outp_in = Variable(torch.from_numpy(np.array([outp_in])))
    outp_out = Variable(torch.from_numpy(outp_out))
    
    if cudaAvailable:
        model = model.cuda()
        inp_t = inp_t.cuda()
        inp_len = inp_len.cuda()
        outp_in = outp_in.cuda()
        outp_out  = outp_out.cuda()
    
    align_score, idxs = model(inp_t, inp_len, outp_in, outp_len)
    align_score = align_score[0].detach().cpu().numpy()
    idxs = np.argmax(align_score, axis=1)
    idxs = idxs.squeeze()
    idxs = PreProcessOutput(idxs)
    
    inp = inp[1:, :]
    ax.scatter(inp[:,0], inp[:, 1])
    # plt.plot(inp[:,0], inp[:,1], 'o')
    for i in range(len(idxs)-1):
        start_pos = inp[idxs[i]]
        end_pos = inp[idxs[i+1]]
        ax.annotate("", xy=start_pos, xycoords='data', xytext=end_pos, textcoords='data',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        # plt.plot(inp[[idxs[i], idxs[i+1]],0], inp[[idxs[i], idxs[i+1]], 1], 'k-')
        

def eval_tour_len_and_acc(align_score):
    idxs = np.argmax(align_score, axis=1)
    print(idxs)
    # idxs = PreProcessOutput(idxs)
    
    
def training(model, train_ds, eval_ds, cudaAvailable, batchSize=10, attention_size=128, beam_width=2, lr=1e-3, clip_norm=2.0,
             weight_decay=0.1, nepoch = 30, model_file="PointerModel.pt", freqEval=5, Teaching_Forcing=1,
             writer=None):
    
  t0 = time()
#  # Pytroch configuration
  if cudaAvailable:
    use_cuda = True
    torch.cuda.device(0)
  else:
    use_cuda = False
  
  train_dl = DataLoader(train_ds, num_workers=0, batch_size=batchSize)
  eval_dl = DataLoader(eval_ds, num_workers=0, batch_size=batchSize)
  
  criterion = PointerNetLoss()
  optimizer = optim.Adam(model.parameters(), lr=lr)
  
  if use_cuda:
    model.cuda()
    
  listOfLoss = []
  listOfLossEval = []
  list_valid_tours = []
  list_valid_tours_eval = []
  cnt = 0
  # Training
  for epoch in range(nepoch):
    model.train()
    total_loss = 0.
    batch_cnt = 0.
    valid_tours = 0
    for b_inp, b_inp_len, b_outp_in, b_outp_out, b_outp_len in train_dl:
      b_inp = Variable(b_inp)
      b_outp_in = Variable(b_outp_in)
      b_outp_out = Variable(b_outp_out)
      if use_cuda:
        b_inp = b_inp.cuda()
        b_inp_len = b_inp_len.cuda()
        b_outp_in = b_outp_in.cuda()
        b_outp_out = b_outp_out.cuda()
        b_outp_len = b_outp_len.cuda()
      
      optimizer.zero_grad()
      align_score, logits, idxs = model(b_inp, b_inp_len, b_outp_in, b_outp_len, Teaching_Forcing=Teaching_Forcing)
      b_outp_len = b_outp_len.squeeze(-1)
      loss = criterion(b_outp_out, align_score, b_outp_len)
      
      # eval_tour_len_and_acc(align_score.detach().cpu().numpy())
      l = loss.item()
      total_loss += l
      batch_cnt += 1
      loss.backward()
      clip_grad_norm(model.parameters(), clip_norm)
      optimizer.step()
      
      idxs = idxs.detach().cpu().numpy()
      
      valid_tours += count_valid_tours(idxs)
          
    writer.add_scalar('training loss', total_loss/batch_cnt, epoch)
    writer.add_scalar('Valid Tours', valid_tours/train_ds.__len__(), epoch)
    
    print("Epoch : {} || loss : {:.3f} || Valid Tours : {:.3f}".format(epoch,
                                                               total_loss / batch_cnt, 
                                                              valid_tours/train_ds.__len__()))
    listOfLoss.append(total_loss/batch_cnt)
    list_valid_tours.append(valid_tours/train_ds.__len__())
    
    if epoch%freqEval==0 and epoch > 0:
        model.eval()
        total_loss_eval = 0
        batch_cnt = 0
        valid_tours_eval = 0
        
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
            
            align_score, logits, idxs = model(b_eval_inp, b_eval_inp_len, b_eval_outp_in, b_eval_outp_len, Teaching_Forcing=0)
            loss = criterion(b_eval_outp_out, align_score, b_eval_outp_len.squeeze(-1))
            l = loss.item()
            total_loss_eval += l
            batch_cnt += 1
            
            idxs = idxs.detach().cpu().numpy()
      
            valid_tours_eval += count_valid_tours(idxs)
        writer.add_scalar('Val loss', total_loss_eval/batch_cnt, epoch)
        writer.add_scalar('Val Valid Tours', valid_tours_eval/train_ds.__len__(), epoch)
    
        print("Epoch: {} || Eval Loss : {:.3f} || Eval Valid Tours : {:.3f}".format(epoch,
                                                                            total_loss_eval/batch_cnt,
                                                                          valid_tours_eval/eval_ds.__len__()))
        listOfLossEval.append(total_loss_eval/batch_cnt)
        list_valid_tours_eval.append(valid_tours_eval/eval_ds.__len__())
  
  # ext. is .pt 
  torch.save(model.state_dict(), model_file)
  t1 = time()
  
  hours, rem = divmod(t1-t0, 3600)
  minutes, seconds = divmod(rem, 60)

  print("Training of Pointer Networks takes: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
  return listOfLoss, listOfLossEval, list_valid_tours, list_valid_tours_eval

if __name__ == "__main__":
    
    train_filename="./CH_TSP_data/tsp5.txt" 
    val_filename = "./CH_TSP_data/tsp5_test.txt"
    cudaAvailable = torch.cuda.is_available()
    
    seq_len = 5
    num_layers = 1
    encoder_input_size = 2 
    rnn_hidden_size = 256
    save_model_name = "PointerModel_Sup_5_teach.pt"
    batch_size = 128
    bidirectional = False
    rnn_type = "LSTM"
    embedding_dim = encoder_input_size # Supervised learning not working w/ embeddings
    attn_type = "Sup"
    C = None
    training_type = "Sup"
    nepoch = 30
    lr = 1e-3
    Teaching_Forcing = 1 #  =1 completamente supervisado
    
    model = PointerNet(rnn_type, bidirectional, num_layers, embedding_dim, rnn_hidden_size, 0, batch_size, attn_type=attn_type, C=C)
    
    weights_init(model)
    
    train_ds = TSPDataset(train_filename, seq_len, training_type, lineCountLimit=-1)
    eval_ds = TSPDataset(val_filename, seq_len, training_type, lineCountLimit=-1)
    
    print("Train data size: {}".format(len(train_ds)))
    print("Eval data size: {}".format(len(eval_ds)))
    
    # Descomentar si es que existe un modelo pre-entrenado.
    # model.load_state_dict(torch.load("PointerModel_Sup_5_sec.pt"))
    
    # Crear summary
    num_exp = 1
    file_writer = "TSP_Sup_" + str(num_exp)
    writer = SummaryWriter('runs/' + file_writer)
    
    
    
    
    # Entrenamiento del modelo
    TrainingLoss, EvalLoss, list_valid_tours, list_valid_tours_eval = training(model, train_ds, eval_ds, cudaAvailable, nepoch=nepoch, 
                                      model_file=save_model_name, batchSize=batch_size, lr=lr, Teaching_Forcing=Teaching_Forcing,
                                      writer=writer)
    # Evaluación del modelo en un conjunto de evaluación
    # eval_model(model, eval_ds, cudaAvailable, n_plt_tours=7, n_cols=3)  