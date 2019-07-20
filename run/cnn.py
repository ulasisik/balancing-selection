#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Convolutional Neural Network(CNN) model and train with images

@author: ulas
"""
import sys 
sys.path.insert(0, '/mnt/NAS/projects/2018_deepLearn_selection/50kb/balancing-selection/scripts') #path to BaSe package

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))

from BaSe.Preprocess import Images
from BaSe.Model import MakeCNN, LoadCNN

test = 2
R = 2700
path_to_images = '/mnt/NAS/projects/2018_deepLearn_selection/50kb/results/images/'  #directory containing input images
path_to_save = '/mnt/NAS/projects/2018_deepLearn_selection/50kb/cnn/model/'  #to save model and data
path_to_figs = '/mnt/NAS/projects/2018_deepLearn_selection/50kb/cnn/figs/'  #directory in which output figures will be saved
image_size = (128, 128)

for selection_start in [10, 15, 20, 25, 30, 35, 40, 'all']:

    #Load images and preprocess:
    images = Images(test, R, image_size, selection_start)
    X_train, X_val, y_train, y_val = images.load_images(path_to_images, verbose=1)
    
    #np.save(path_to_save+"X_train_"+str(test), X_train, allow_pickle=False)
    #np.save(path_to_save+"y_train"+str(test), y_train, allow_pickle=False)
    #np.save(path_to_save+"X_val_"+str(test), X_val, allow_pickle=False)
    #np.save(path_to_save+"y_val_"+str(test), y_val, allow_pickle=False)
    
    # CNN
    epoch = 20
    batch_size = 64

    #Load tuned and compiled model 
    mymodel = LoadCNN(test)

    print(mymodel.model.summary())

    #Fit model 
    mymodel.fit_model(X_train, y_train, batch_size, epoch, validation_data = (X_val, y_val),
                    file_model = None, file_hist = path_to_save+'CNN_hist_'+str(selection_start)+'_'+str(test)+'.txt')

    #Plot loss, accuracy and confusion matrix
    #mymodel.vis_acc_loss(path_to_figs+'CNN_loss_acc.eps')
    #mymodel.vis_cm(X_test, y_test, file_fig = path_to_figs+'CNN_cm.eps', file_mat = None)
