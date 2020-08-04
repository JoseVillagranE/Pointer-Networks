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

from TSPDataset import TSPDataset
from pointer_network import PointerNet, PointerNetLoss
import warnings
warnings.filterwarnings("ignore")

'''
TODO:
    - implementar el largo del viaje como metrica a evaluar
    - evaluar de forma visual en x viajes a modo de testing
    - implementar un formato de evaluación supervisado
    - Mecanismo de predicción sin label. (test time)
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
    
    outp = np.array([outp[i] - 1 for i in range(outp.shape[0]) if outp[i] != 0])
    return outp


def eval_model(model, eval_ds, cudaAvailable, batchSize=1):
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
        
        align_score = model(b_eval_inp, b_eval_inp_len, b_eval_outp_in, b_eval_outp_len, Mode="Eval")
        align_score = align_score.cpu().detach().numpy()
        idxs = np.argmax(align_score, axis=2)
        b_eval_outp_out = b_eval_outp_out.cpu().detach().numpy()
        b_eval_outp_out = b_eval_outp_out.squeeze()
        idxs = PreProcessOutput(idxs.squeeze(0))
        labels = PreProcessOutput(b_eval_outp_out)
        if functools.reduce(lambda i, j: i and j, map(lambda m, k: m==k, idxs, labels), True):
            # Evaluación estricta
            countAcc += 1
    
    Acc = countAcc/eval_ds.__len__()
    print("The Accuracy of the model is: {}".format(Acc))


def plot_one_tour(model, example, cudaAvailable):
    
    model.eval()
    inp, inp_len, outp_in, outp_out, outp_len = example
    inp_t = Variable(torch.from_numpy(np.array([inp])))
    inp_len = torch.from_numpy(inp_len)
    outp_in = Variable(torch.from_numpy(np.array([outp_in])))
    outp_out = Variable(torch.from_numpy(outp_out))
    align_score = model(inp_t, inp_len, outp_in, outp_len)
    align_score = align_score[0].detach().numpy()
    idxs = np.argmax(align_score, axis=1)
    idxs = PreProcessOutput(idxs)
    
    inp = inp[1:, :]
    plt.figure(figsize=(10,10))
    plt.plot(inp[:,0], inp[:,1], 'o')
    for i in range(idxs.shape[0]-1):
        plt.plot(inp[[idxs[i], idxs[i+1]],0], inp[[idxs[i], idxs[i+1]], 1], 'k-')
        

def eval_tour_len_and_acc(align_score):
    idxs = np.argmax(align_score, axis=1)
    print(idxs)
    # idxs = PreProcessOutput(idxs)
    
    
def training(model, train_ds, eval_ds, cudaAvailable, batchSize=10, attention_size=128, beam_width=2, lr=1e-3, clip_norm=2.0,
             weight_decay=0.1, nepoch = 30, model_file="PointerModel.pt", freqEval=5):
    
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
  cnt = 0
  # Training
  for epoch in range(nepoch):
    model.train()
    total_loss = 0.
    batch_cnt = 0.
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
      align_score = model(b_inp, b_inp_len, b_outp_in, b_outp_len)
      b_outp_len = b_outp_len.squeeze(-1)
      loss = criterion(b_outp_out, align_score, b_outp_len)
      
      # eval_tour_len_and_acc(align_score.detach().cpu().numpy())
      l = loss.item()
      total_loss += l
      batch_cnt += 1

      loss.backward()
      clip_grad_norm(model.parameters(), clip_norm)
      optimizer.step()
    print("Epoch : {}, loss {}".format(epoch, total_loss / batch_cnt))
    listOfLoss.append(total_loss/batch_cnt)
    
    if(epoch%freqEval==0):
        model.eval()
        total_loss_eval = 0
        batch_cnt = 0
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
            
            align_score = model(b_eval_inp, b_eval_inp_len, b_eval_outp_in, b_eval_outp_len)
            loss = criterion(b_eval_outp_out, align_score, b_eval_outp_len.squeeze(-1))
            l = loss.item()
            total_loss_eval += l
            batch_cnt += 1
        print("Epoch: {}, Eval Loss {}".format(epoch, total_loss_eval/batch_cnt))
        listOfLossEval.append(total_loss_eval/batch_cnt)
  
  # ext. is .pt 
  torch.save(model.state_dict(), model_file)
  t1 = time()
  
  hours, rem = divmod(t1-t0, 3600)
  minutes, seconds = divmod(rem, 60)

  print("Training of Pointer Networks takes: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
  return listOfLoss, listOfLossEval

if __name__ == "__main__":
    
    train_filename="./CH_TSP_data/tsp5.txt" 
    val_filename = "./CH_TSP_data/tsp5_test.txt"
    cudaAvailable = torch.cuda.is_available()
    
    seq_len = 5
    num_layers = 1
    encoder_input_size = 2 
    rnn_hidden_size = 32
    save_model_name = "PointerModel_Sup_5.pt"
    batch_size = 10000
    bidirectional = False
    rnn_type = "LSTM"
    embedding_dim = encoder_input_size # Supervised learning not working w/ embeddings
    attn_type = "Sup"
    C = None
    training_type = "Sup"
    nepoch = 20
    
    model = PointerNet(rnn_type, bidirectional, num_layers, embedding_dim, rnn_hidden_size, 0, batch_size, attn_type=attn_type, C=C)
    
    weights_init(model)
    
    train_ds = TSPDataset(train_filename, seq_len, training_type, lineCountLimit=100)
    eval_ds = TSPDataset(val_filename, seq_len, training_type, lineCountLimit=100)
    
    print("Train data size: {}".format(len(train_ds)))
    print("Eval data size: {}".format(len(eval_ds)))
    
    # Descomentar si es que existe un modelo pre-entrenado.
    # model.load_state_dict(torch.load("PointerModel_Sup_5.pt"))
    
    # Entrenamiento del modelo
    TrainingLoss, EvalLoss = training(model, train_ds, eval_ds, cudaAvailable, nepoch=nepoch, 
                                      model_file=save_model_name, batchSize=batch_size)
    # Evaluación del modelo en un conjunto de evaluación
    eval_model(model, eval_ds, cudaAvailable)

    # Grafica un viaje de ejemplo
    # example = eval_ds.__getitem__(0)
    # plot_one_tour(model, example, cudaAvailable)    