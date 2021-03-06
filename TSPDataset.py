# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:12:13 2020

@author: joser
"""

import numpy as np
from torch.utils.data import Dataset
from copy import copy # shallow copy. copy dont change the old list
from utils import get_batch_nodes

class TSPDataset(Dataset):
    
    def __init__(self, filename, f_city_fixed=True, lineCountLimit=-1):
        super().__init__()
        
        self.lineCountLimit = lineCountLimit
        self.f_city_fixed = f_city_fixed
        self.START = [0, 0] # Token
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
                
                
                
                outp_out = []
                
                inp_len = len(inp)
                outp_len = len(outp)
                
                cnt = 0
                idxs = []
                for idx in outp:
                    if not idx in idxs: 
                        if cnt==0:
                            outp_in = inp[2*(idx - 1): 2*idx]
                        else:
                            outp_in +=  inp[2*(idx - 1): 2*idx]
                        cnt+=1
                        idxs.append(idx)
                    outp_out += [idx]
                    
                if self.f_city_fixed:
                    inp = self.START + inp
                    inp_len += 1
                    outp_in = self.START + outp_in
                    # outp_in = outp_in
                
                inp_len = len(inp) // 2
                
                inp = np.array(inp).reshape([-1, 2])
                inp_len = np.array([inp_len])
                
                
                outp_in = np.array(outp_in).reshape([-1, 2])
                
                outp_out = outp_out[:-1] 
                outp_len -= 1
                # outp_in = outp_in[:-1]
                
                if self.f_city_fixed:
                    outp_out = [0] + outp_out + [0]
                    outp_len += 2
                else:
                    # outp_out += [outp_out[0]]
                    outp_out = outp_out - np.ones_like(outp_out)
                    # outp_len += 1
                
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



class Generator(Dataset):
	def __init__(self, n_samples, city_t, seed = None):
		self.data = get_batch_nodes(n_samples, city_t, seed=seed)
		
	def __getitem__(self, idx):
		return self.data[idx]

	def __len__(self):
		return self.data.size(0)

if __name__=="__main__":
    train_ds = TSPDataset("./CH_TSP_data/tsp5.txt", f_city_fixed=True, lineCountLimit=5)
    
    inp, inp_len, outp_in, outp_out, outp_len = train_ds.__getitem__(0)
    
    print(inp)
    print(outp_out)
    print(outp_in)
    print(inp_len)
    print(outp_len)
    
    train_ds = TSPDataset("./CH_TSP_data/tsp5.txt", f_city_fixed=False, lineCountLimit=5)
    
    inp, inp_len, outp_in, outp_out, outp_len = train_ds.__getitem__(0)
    
    print(inp)
    print(outp_out)
    print(outp_in)
    print(inp_len)
    print(outp_len)
                    
    