#!/usr/bin/env python3.6
"""
Create Artificial Neural Network(ANN) model and train with summary statistics.

@author: ulas isildak
"""
import sys
sys.path.insert(0, '/mnt/NAS/projects/2018_deepLearn_selection/50kb/balancing-selection/scripts') #path to BaSe package
from BaSe.Model import MakeANN, LoadANN

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))

test = 1  #test number to be performed (1,2, or 3)
R = 900  #number of iterations to include
path_to_stats = '/mnt/NAS/projects/2018_deepLearn_selection/50kb/ann/' #path to summary statistics
path_to_figs = '/mnt/NAS/projects/2018_deepLearn_selection/50kb/ann/figs/' #location where figs will be saved
path_to_save = '/mnt/NAS/projects/2018_deepLearn_selection/50kb/ann/model/' # directory in which the trained model will be saved

for selection_start in [10, 15, 20, 25, 30, 35, 40, 'all']:
    # Data Import & Preprocessing
    stats = SumStats(test, R, selection_start)
    X_train, X_val, y_train, y_val = stats.load_sumstats(path_to_stats)
    
    # ANN
    epoch = 20
    batch_size = 64

    #Load tuned & compiled model
    mymodel = LoadANN(test)

    print(mymodel.model.summary())

    #Fit model
    mymodel.fit_model(X_train, y_train, batch_size, epoch, validation_data = (X_val, y_val),
                    file_model = None, file_hist = path_to_save+'ANN_hist_'+str(selection_start)+'_'+str(test)+'.txt')

    #Plot loss, accuracy and confusion matrix
    #mymodel.vis_acc_loss(path_to_figs+'ANN_loss_acc.eps')
    #mymodel.vis_cm(X_test, y_test, file_fig = path_to_figs+'ANN_cm.eps', file_mat = None)
