#!/usr/bin/env python3
"""
Calculates summary statistics from simulations

@author: ulas isildak
"""

import sys
sys.path.insert(0, '/mnt/NEOGENE1/projects/deepLearn_selection_2018/balancing-selection')

from BaSe.Preprocess import sum_stat

path_to_sim = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/results/decompMSMS/'
path_to_stats = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/results/stats/'

N = 50000  # length of simulated sequence (selection scenarios)
N_NE = 500000 # length of simulated sequence (neutral)
NCHROMS = 198
REP_TO = 1000
REP_FROM = 1

for cls in ["NE", "IS", "OD", "FD"]:
    sum_stat(path_to_sim, path_to_stats, cls, NCHROMS, REP_FROM, REP_TO, N, N_NE)