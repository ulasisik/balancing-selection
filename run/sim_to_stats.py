#!/usr/bin/env python3.6
"""
Calculates summary statistics from simulations

@author: ulas isildak
"""

import sys
sys.path.insert(0, '/mnt/NAS/projects/2018_deepLearn_selection/50kb/balancing-selection/scripts') #path to BaSe package

from BaSe.Preprocess import sum_stat

path1='/mnt/NEOGENE1/projects/deepLearn_selection_2018/50kb/results/decompMSMS/'
path2='/mnt/NEOGENE1/projects/deepLearn_selection_2018/50kb/ann/'

N = 50000  # length of simulated sequence (selection scenarios)
N_NE = 500000 # length of simulated sequence (neutral)
NCHROMS = 198
REP_TO = 2000
REP_FROM = 4000
cl = 'OD'  #NE, IS, OD, or FD

sum_stat(path1, path2, cl, NCHROMS, REP_TO, REP_FROM, N, N_NE)