# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:12:13 2020

@author: joser
"""

import numpy as np
from torch.utils.data import Dataset
from copy import copy # shallow copy. copy dont change the old list

class TSPDataset(Dataset):
    
    def __init__(self, filename, seq_len, lineCountLimit=-1):
        super().__init__()
        
        self.seq_len = seq_len
        self.lineCountLimit = lineCountLimit
        self.START = [0, 0]
        self.END = [0, 0]
        self.load_data(filename)
        
    
    def load_data(self, filename):
        
        with open(filename, 'r') as f:
            lineCount = 0
            data = []
            for line in f:
                
                if lineCount == self.lineCountLimit:
                    break
                
                inp, outp = line.strip().split('output')
                
                inp = list(map(float, inp.strip().split(' '))) # input -> string
                outp = list(map(int, outp.strip().split(' ')))
                
                # outp_in = copy(self.START)
                outp_out = []
                
                cnt = 0
                for idx in outp:
                    if cnt==0:
                        outp_in = inp[2*(idx - 1): 2*idx]
                    else:
                        outp_in +=  inp[2*(idx - 1): 2*idx]
                    outp_out += [idx]
                    cnt+=1
                
                # outp_out += [0] # token
                
                inp_len = len(inp) // 2
                
                #inp = self.START + inp
                #inp_len += 1
                assert self.seq_len + 1 >= inp_len
                # for i in range(self.seq_len + 1 - inp_len):
                #     inp += self.END
                
                inp = np.array(inp).reshape([-1, 2])
                inp_len = np.array([inp_len])
                outp_len = len(outp) + 1
                
                # for i in range(self.seq_len + 2 - outp_len):
                #     outp_in += self.START
                
                outp_in = np.array(outp_in).reshape([-1, 2])
                outp_out = outp_out + [0] * (self.seq_len + 2 - outp_len)
                outp_out = np.array(outp_out)
                outp_len = np.array([outp_len])
                lineCount += 1
            
                data.append((inp.astype("float32"), inp_len, outp_in.astype("float32"), outp_out, outp_len))
            self.data = data
                
                
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        inp, inp_len, outp_in, outp_out, outp_len = self.data[index]
        return inp, inp_len, outp_in, outp_out, outp_len            
                

if __name__=="__main__":
    train_ds = TSPDataset("./CH_TSP_data/tsp5.txt", 5, lineCountLimit=5)
    
    inp, inp_len, outp_in, outp_out, outp_len = train_ds.__getitem__(0)
    
    print(inp)
    print(outp_out)
    print(outp_in)
                    
    