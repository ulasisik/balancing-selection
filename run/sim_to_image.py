#!/usr/bin/env python3.6
"""
Creates images from simulations

@author: ulas isildak
"""

import sys
sys.path.insert(0, '/Users/ulas/Projects/BS_DeepLearning/balancing-selection')

from BaSe.Preprocess import sim_to_image, image_to_array

path_to_sim = '/Users/ulas/Projects/BS_DeepLearning/raw_data/'
path_to_images = '/Users/ulas/Projects/BS_DeepLearning/images/'
path_to_arrays = '/Users/ulas/Projects/BS_DeepLearning/arrays/'

N = 50000  #length of simulated sequence (selection scenarios)
N_NE = 500000 #length of simulated sequence (neutral)
NCHROMS = 198
REP_TO = 100
REP_FROM = 1
img_dim = (128,128)
clss = ("NE", "IS", "OD", "FD")


sim_to_image(path_to_sim, path_to_images, REP_FROM, REP_TO, NCHROMS, N, N_NE, img_dim, clss)

image_to_array(path_to_images, path_to_arrays, REP_TO, img_dim)