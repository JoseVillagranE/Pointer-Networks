# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:53:35 2020

@author: joser
"""
import numpy as np
import math

def compute_len_tour(tour, idxs):
    """
    Parameters
    ----------
    tour : Numpy Array
        Nodes of Tour
    idxs : Python list
        ordered idxs by pointer

    Returns
    
    tour_len: float
    -------
    """
    len_tour = 0
    for i in range(tour.shape[1] - 1):
        len_tour += np.linalg.norm(tour[:, i, :] - tour[:, i+1, :], axis=1)
    # Back to home
    len_tour += np.linalg.norm(tour[:, -1, :] - tour[:, 0, :], axis=1)
    return len_tour[0]

def count_valid_tours(idxs, axis=1):
    
    """
    idxs -> [seq_len, batch]
    """
    valid_tours = 0
    for i in range(idxs.shape[1]):
        
        idx_i = idxs[:, i]
        idx_i_unique = np.unique(idx_i)
        if idx_i.shape[0] == idx_i_unique.shape[0] + 1 and idx_i[0] == idx_i[-1]:
            valid_tours += 1
    return valid_tours

def name_creation(filetype="pt", *args):
    
    if filetype=="pt":
        name = "PointerModel_"
    elif filetype == "txt":
        name = "logs_"
    else:
        print("The filetype especified is not implemented")
        return None
    
    for arg in args:
        if type(arg) == str:
            name += arg
        else:
            name += str(arg)
        
        if not arg == args[-1]:
            name += "_"
        else:
            name += "." + filetype
    return name
        
def logs_sup_training(tr_loss, val_loss, valid_tr, valid_val, freq_eval, *args):
    
    file_name = name_creation("txt", *args)
    f = open(file_name, "w+")
    j = 0
    for i in range(len(tr_loss)):
        f.write("Epoch : {} || loss : {:.3f} || Valid Tours : {:.3f}\n".format(i,
                                                               tr_loss[i],
                                                              valid_tr[i]))
        if i%freq_eval:
            f.write("Epoch : {} || Val_loss : {:.3f} || Val_Valid Tours : {:.3f}\n".format(i,
                                                               val_loss[j], 
                                                              valid_val[j]))
            j += 1
            
    f.close()

# ref: https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
def beam_search_decoder(probs, beam_width=3):
    
    eps = 1e-7
    sequences = [[list(), 0.0]]
    # walk over each step in sequence
    for row in probs:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                prob = row[j]
                if row[j] < 1e-15:
                    prob = row[j] + eps
                if j not in seq:
                    candidate = [seq + [j], score - math.log(prob)]
                    all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:beam_width]
    return sequences


if __name__ == "__main__":
    
    # ------------------- Debug for compute_len_tour and count valid tours ----------------
    
    tour = np.random.rand(1, 5, 2)
    print("Tour:", tour)
    
    idxs = [0, 3, 2, 1, 4]
    
    len_tour = compute_len_tour(tour, idxs)
    print("Len tour: ", len_tour)
    
    idxs = np.array([[1, 1, 1, 1], [1, 2, 3, 1], [1, 2, 2, 3]])
    print(idxs)
    
    valid_tours = count_valid_tours(idxs)
    print("Valid Tours: ", valid_tours)
    
    # -------------------------------------------------------------------------------------
    
    # name_pt = name_creation("pt", "Sup", 5, 128)
    # name_txt = name_creation("txt", "Sup",  5, 128)
    # print(name_pt)
    # print(name_txt)
    
    # -----------------------Debug for logs ------------------------------------------------------
    
    # tr_loss= [0.6, 0.3, 0.23, 0.1, 0.3] 
    # val_loss = [0.34, 0.15]
    # valid_tr = [0.4, 0.6, 0.7, 0.78, 0.8]
    # valid_val = [0.65, 0.78]
    # freq_eval = 2
    
    # logs_sup_training(tr_loss, val_loss, valid_tr, valid_val, freq_eval, "test")
    
    # -------------------------Debug for Beam Search -----------------------------------------------
    
#     data = [[1e-55, 0.2, 0.3, 0.4, 0.5],
# 		[0.1, 0.2, 0.3, 0.4, 0.5],
# 		[0.5, 0.4, 0.3, 0.2, 0.1]]
#     data = np.array(data)
#     # decode sequence
#     result = beam_search_decoder(data, 3)
#     # print result
#     for seq in result:
#     	print(seq)
    