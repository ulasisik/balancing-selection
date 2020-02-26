#!/usr/bin/env python3
"""
Creates images from simulations

@author: ulas isildak
"""

import sys
sys.path.insert(0, '/mnt/NEOGENE1/projects/deepLearn_selection_2018/balancing-selection')

from BaSe.Preprocess import sim_to_image

path_to_sim = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/results/decompMSMS/'
path_to_images = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/results/images/'

N = 50000       #length of simulated sequence (selection scenarios)
N_NE = 500000   #length of simulated sequence (neutrality)
NCHROMS = 198
REP_TO = 1000
REP_FROM = 1
img_dim = (128,128)
clss = ("IS", "OD", "FD", "NE")


sim_to_image(path_to_sim, path_to_images, REP_FROM, REP_TO, NCHROMS, N, N_NE, img_dim, clss, sort="freq", method="s")
