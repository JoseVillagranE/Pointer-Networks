# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:53:35 2020

@author: joser
"""
import numpy as np



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
    
    
    valid_tours = 0
    for i in range(idxs.shape[0]):
        idx_i = idxs[i, :]
        idx_i_unique = np.unique(idx_i)
        if idx_i.shape[0] == idx_i_unique.shape[0]:
            valid_tours += 1
    return valid_tours
        



if __name__ == "__main__":
    
    tour = np.random.rand(1, 5, 2)
    print("Tour:", tour)
    
    idxs = [0, 3, 2, 1, 4]
    
    len_tour = compute_len_tour(tour, idxs)
    print("Len tour: ", len_tour)
    
    idxs = np.array([[1, 1, 1], [1, 2, 3], [1, 2, 2]])
    
    valid_tours = count_valid_tours(idxs)
    print("Valid Tours: ", valid_tours)
    