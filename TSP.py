# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 23:52:01 2020

@author: joser
"""

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import sys
import matplotlib.pyplot as plt

import functools
from time import time

from TSPDataset import TSPDataset
#sys.path.insert(0, "D:/MaterialU/12 semestre/Trabajo_dirigido/Diego/codigo/PointerNetwork")
from pointer_network import PointerNet, PointerNetLoss


def training(model, train_ds, eval_ds, cudaAvailable, batchSize=1, attention_size=128, beam_width=2, lr=1e-3, clip_norm=5.0,
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
      loss = criterion(b_outp_out, align_score, b_outp_len)

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
            loss = criterion(b_eval_outp_out, align_score, b_eval_outp_len)
            l = loss.item()
            total_loss_eval += l
            batch_cnt += 1
        print("Epoch: {}, Eval Loss {}".format(epoch, total_loss_eval/batch_cnt))
        listOfLossEval.append(total_loss_eval/batch_cnt)
  
  # ext. is .pt 
  torch.save(model.state_dict(), model_file)
  t1 = time()
  print("Training of Pointer Network takes: {}".format(t1-t0))
  return listOfLoss, listOfLossEval

if __name__ == "__main__":
    
    train_filename="./data/tsp5.txt" 
    val_filename = "./data/tsp5_test.txt"
    cudaAvailable = torch.cuda.is_available()
#    cudaAvailable = False
    seq_len = 5
    num_layers = 1
    rnn_hidden_size = 32
    
    model = PointerNet("LSTM", True, num_layers, 2, rnn_hidden_size, 0.0)
    
    train_ds = TSPDataset(train_filename, seq_len)
    eval_ds = TSPDataset(val_filename, seq_len)
    
    print("Train data size: {}".format(len(train_ds)))
    print("Eval data size: {}".format(len(eval_ds)))
    
#    model.load_state_dict(torch.load("PointerModel.pt"))
    training(model, train_ds, eval_ds, cudaAvailable, nepoch=100)
#    eval_model(model, train_ds, cudaAvailable)    
#    example = train_ds.__getitem__(0)