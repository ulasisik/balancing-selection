#!/usr/bin/env python3
"""
Creates images from simulations

@author: ulas isildak
"""

import sys
sys.path.insert(0, '/Users/ulas/Projects/balancing_selection')
from BaSe.Preprocess import sim_to_image

path_to_sim = '/Users/ulas/Projects/balancing_selection/Data/SimOuts/'
path_to_image = '/Users/ulas/Projects/balancing_selection/Data/Images/'

clss = ("NE", "IS", "OD", "FD")
N = 50000
N_NE = 500000
NCHROMS = 198
SIM_FROM = 1
SIM_TO = 8000
img_dim = (128, 128)

sim_to_image(path_to_sim, path_to_image, SIM_FROM, SIM_TO, NCHROMS, N, N_NE, img_dim)
