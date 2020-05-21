#!/usr/bin/env python3
"""
Create Artificial Neural Network(ANN) model and train with summary statistics.

@author: ulas isildak
"""

import sys
sys.path.insert(0, '/Users/ulas/Projects/balancing_selection')
from BaSe.Preprocess import SumStats
from BaSe.Model import LoadANN

path_to_stat = '/Users/ulas/Projects/balancing_selection/Data/Stats/'
path_to_save = '/Users/ulas/Projects/balancing_selection/Data/Model/'
path_to_figs = '/Users/ulas/Projects/balancing_selection/Results/Figures/'

N = 20000

for test in [1, 2, 3]:
    for selection_category in ["recent", "medium", "old"]:
        stats = SumStats(test, selection_category, N)
        X_train, X_val, y_train, y_val = stats.load_sumstats(path_to_stat, val_size=0.2, toshuffle=True, scale=True,
                                                             pca=True, random_state=2, verbose=1)

        mymodel = LoadANN(test, input_shape=66)
        mymodel.summary()

        mymodel.fit_model(X_train, y_train, batch_size=32, epoch=10, validation_data=(X_val, y_val), verbose=2)

        mymodel.save_model("{}ANN_model_{}_{}.h5".format(path_to_save, selection_category, test))

        mymodel.vis_acc_loss(file="{}ANN_lossAcc_{}_{}.eps".format(path_to_figs, selection_category, test))
        mymodel.vis_cm(X_val, y_val, stats.classes,
                       file_fig="{}ANN_cm_{}_{}.eps".format(path_to_figs, selection_category, test))
