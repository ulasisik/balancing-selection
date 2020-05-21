#!/usr/bin/env python3
"""
Calculates summary statistics from simulations

@author: ulas isildak
"""
import sys
sys.path.insert(0, '/Users/ulas/Projects/balancing_selection')
from BaSe.Preprocess import sim_to_stats

path_to_sim = '/Users/ulas/Projects/balancing_selection/Data/SimOuts/'
path_to_stat = '/Users/ulas/Projects/balancing_selection/Data/Stats/'

N = 50000
N_NE = 500000
NCHROMS = 198
SIM_FROM = 1
SIM_TO = 8000

for clss in ["NE", "IS", "OD", "FD"]:
    sim_to_stats(path_to_sim, path_to_stat, clss, NCHROMS, SIM_FROM, SIM_TO, N, N_NE)
