#!/usr/bin/env python3.6
"""
Calculates summary statistics from simulations

@author: ulas isildak
"""

import sys
sys.path.insert(0, '/Users/ulas/Projects/BS_DeepLearning/balancing-selection')

from BaSe.Preprocess import sum_stat

path_to_sim = '/Users/ulas/Projects/BS_DeepLearning/raw_data/'
path_to_stats = '/Users/ulas/Projects/BS_DeepLearning/stats/'

N = 50000  # length of simulated sequence (selection scenarios)
N_NE = 500000 # length of simulated sequence (neutral)
NCHROMS = 198
REP_TO = 10
REP_FROM = 1

for cls in ["NE", "IS", "OD", "FD"]:
    sum_stat(path_to_sim, path_to_stats, cls, NCHROMS, REP_FROM, REP_TO, N, N_NE)