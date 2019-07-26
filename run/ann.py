#!/bin/usr/env python3
"""
Create Artificial Neural Network(ANN) model and train with summary statistics.

@author: ulas isildak
@e-mail: isildak.ulas [at] gmail.com
"""
import sys
sys.path.insert(0, '/mnt/NEOGENE1/projects/deepLearn_selection_2018/balancing-selection') #path to BaSe package
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))

from BaSe.Preprocess import SumStats
from BaSe.Model import MakeANN, LoadANN

path_to_stats = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/results/stats/'
path_to_figs = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/ann/figs/'
path_to_save = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/ann/model/'

selection_start = 20    # Selection start time: 20, 25, 30, 35, 40, or 'all'
test = 1    # Test number: 1, 2, or 3
R = 5000  # Number of samples


# Data import & preprocessing
stats = SumStats(test, R, selection_start)
X_train, X_val, y_train, y_val = stats.load_sumstats(path_to_stats)

# ANN
epoch = 50
batch_size = 64

# Load tuned & compiled model
mymodel = LoadANN(test)

print(mymodel.model.summary())

# Fit model
mymodel.fit_model(X_train, y_train, batch_size, epoch, validation_data = (X_val, y_val),
                file_model = "{}ANN_model_{}_{}.h5".format(path_to_save, selection_start, test), 
                file_hist = "{}ANN_hist__{}_{}.txt".format(path_to_save, selection_start, test))

# Plot loss & accuracy plot and confusion matrix
mymodel.vis_acc_loss("{}ANN_lossAcc_{}_{}.eps".format(path_to_save, selection_start, test))
mymodel.vis_cm(X_val, y_val, 
        file_fig = "{}ANN_cm_{}_{}.eps".format(path_to_save, selection_start, test), 
        file_mat = "{}ANN_cm_{}_{}.txt".format(path_to_save, selection_start, test))

