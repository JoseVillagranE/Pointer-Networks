# coding=utf-8

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import sys
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import functools
from time import time

from dataset import CHDataset
sys.path.insert(0, "D:/matricula u chile 2015/12 semestre/Trabajo_dirigido/Diego/codigo/PointerNetwork-PyTorch-master")
from pointer_network import PointerNet, PointerNetLoss


def PreProcessOutput(outp):
    
    outp = np.array([outp[i] - 1 for i in range(outp.shape[0]) if outp[i] != 0])
    return outp
    
def plot_convex_hull(inp, labels):
    
    inp = inp[1:, :]
    print(labels.shape)
    labels = np.array([labels[i] - 1 for i in range(labels.shape[0]) if labels[i] != 0])
    
    hull = ConvexHull(inp)
    
    plt.figure(figsize=(10,10))
    plt.plot(inp[:,0], inp[:,1], 'o')
    
    print(hull.simplices)
    print(labels)
    
    for simplex in hull.simplices:
        plt.plot(inp[simplex, 0], inp[simplex, 1], 'k-')
        
    plt.title("Convex Hull from scipy")
    plt.figure(figsize=(10,10))
    plt.plot(inp[:,0], inp[:,1], 'o')
    
    print(labels.shape[0])
    for i in range(labels.shape[0]-1):
        plt.plot(inp[[labels[i], labels[i+1]],0], inp[[labels[i], labels[i+1]], 1], 'k-')
    
    plt.title("Convex Hull from predict-labels")
    

def eval_one_example(model, example):
    
    model = model.eval().cpu()
    
    model.eval()
    inp, inp_len, outp_in, outp_out, outp_len = example
    inp_t = Variable(torch.from_numpy(np.array([inp])))
    inp_len = torch.from_numpy(inp_len)
    outp_in = Variable(torch.from_numpy(np.array([outp_in])))
    outp_out = Variable(torch.from_numpy(outp_out))
    align_score = model(inp_t, inp_len, outp_in, outp_len)
    align_score = align_score.detach().numpy()
    idxs = np.argmax(align_score[0], axis=1)
    idxs = PreProcessOutput(idxs)
    
    inp = inp[1:, :]
    plt.figure(figsize=(10,10))
    plt.plot(inp[:,0], inp[:,1], 'o')
    for i in range(idxs.shape[0]-1):
        plt.plot(inp[[idxs[i], idxs[i+1]],0], inp[[idxs[i], idxs[i+1]], 1], 'k-')
        

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
        
        align_score = model(b_eval_inp, b_eval_inp_len, b_eval_outp_in, b_eval_outp_len)
        align_score = align_score[0].cpu().detach().numpy()
        idxs = np.argmax(align_score, axis=1)
        b_eval_outp_out = b_eval_outp_out.cpu().detach().numpy()
        b_eval_outp_out = b_eval_outp_out.squeeze()
        idxs = PreProcessOutput(idxs)
        labels = PreProcessOutput(b_eval_outp_out)
        
        if functools.reduce(lambda i, j: i and j, map(lambda m, k: m==k, idxs, labels), True):
            countAcc += 1
    
    Acc = countAcc/eval_ds.__len__()
    print("The Accuracy of the model is: {}".format(Acc))
    
    
        

def PlotLossCurve(Loss, title, shape = (10, 10)):
    
#    plt.figure(shape)
    
    plt.plot(range(len(Loss)), np.array(Loss))
    plt.ylabel("Loss", fontsize=10)
    plt.xlabel("Epoch*Num_Batch", fontsize=10)
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()
    
    
    



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
      align_score, _, _ = model(b_inp, b_inp_len, b_outp_in, b_outp_len)
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
            
            align_score, _, _ = model(b_eval_inp, b_eval_inp_len, b_eval_outp_in, b_eval_outp_len)
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
    
    train_filename="./data/convex_hull_50_train.txt" 
    val_filename = "./data/convex_hull_50_test.txt"
    cudaAvailable = torch.cuda.is_available()
    max_in_seq_len=50
    max_out_seq_len=11
    num_layers = 1
    rnn_hidden_size = 32
    save_model_file = "ConvexHull.pt"
    
    model = PointerNet("LSTM", True, num_layers, 2, rnn_hidden_size, 0.0)
    
    train_ds = CHDataset(train_filename, max_in_seq_len, max_out_seq_len, 1000)
    eval_ds = CHDataset(val_filename, max_in_seq_len, max_out_seq_len, 50)
    
    print("Train data size: {}".format(len(train_ds)))
    print("Eval data size: {}".format(len(eval_ds)))
    
    # Descomentar si existe algún modelo
    # model.load_state_dict(torch.load("PointerModel.pt"))
    
    # Entrenamiento
    
    list_Loss, Loss_eval = training(model, train_ds, eval_ds, cudaAvailable, nepoch=100, model_file = save_model_file)

    # Evaluación del modelo en un conjunto de evaluación
    eval_model(model, train_ds, cudaAvailable)    
  
    #    title="Trainning_Loss"
    #    PlotLossCurve(list_Loss, title, shape = (10, 10))

    
    