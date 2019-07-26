#!/usr/bin/env python3
"""
Create Convolutional Neural Network(CNN) model and train with images

@author: ulas isildak
@e-mail: isildak.ulas [at] gmail.com
"""
import sys 
sys.path.insert(0, '/mnt/NEOGENE1/projects/deepLearn_selection_2018/balancing-selection') #path to BaSe package

import numpy as np
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))

from BaSe.Preprocess import Images
from BaSe.Model import LoadCNN

path_to_images = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/results/images/' 
path_to_save = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/cnn/model/' 
path_to_figs = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/cnn/figs/' 

selection_start = 20    # Selection start time: 20, 25, 30, 35, 40, or 'all'
image_size = (128, 128)
test = 1    # Test number: 1, 2, or 3
R = 5000    # Number of samples

# Load images and preprocess:
images = Images(test, R, image_size, selection_start)
X_train, X_val, y_train, y_val = images.load_images(path_to_images)

#np.save("{}X_train_{}_{}".format(path_to_save, selection_start, test), X_train, allow_pickle=False)
#np.save("{}y_train_{}_{}".format(path_to_save, selection_start, test), y_train, allow_pickle=False)
#np.save("{}X_val_{}_{}".format(path_to_save, selection_start, test), X_val, allow_pickle=False)
#np.save("{}y_val_{}_{}".format(path_to_save, selection_start, test), y_val, allow_pickle=False)

# CNN
epoch = 50
batch_size = 64

#Load tuned and compiled model 
mymodel = LoadCNN(test, selection_start)

print(mymodel.model.summary())

# Fit model 
mymodel.fit_model(X_train, y_train, batch_size, epoch, aug = True, validation_data = (X_val, y_val),
                file_model = "{}CNN_model_{}_{}.h5".format(path_to_save, selection_start, test), 
                file_hist = "{}CNN_hist__{}_{}.txt".format(path_to_save, selection_start, test))

# Plot loss & accuracy plot and confusion matrix
mymodel.vis_acc_loss("{}CNN_lossAcc_{}_{}.eps".format(path_to_save, selection_start, test))
mymodel.vis_cm(X_val, y_val, 
        file_fig = "{}CNN_cm_{}_{}.eps".format(path_to_save, selection_start, test), 
        file_mat = "{}CNN_cm_{}_{}.txt".format(path_to_save, selection_start, test))

