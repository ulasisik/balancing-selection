#!/usr/bin/env python3
"""
Create Convolutional Neural Network(CNN) model and train with images

@author: ulas isildak
"""

import sys 
sys.path.insert(0, '/mnt/NEOGENE1/projects/deepLearn_selection_2018/balancing-selection')

import numpy as np
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))

from BaSe.Preprocess import Images
from BaSe.Model import LoadCNN

path_to_images = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/results/images/' 
path_to_save = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/cnn/model/' 
path_to_figs = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/cnn/figs/' 

selection_category = "recent"   # Time of onset of selection
image_size = (128, 128)
test = 1                        # Test number: 1, 2, or 3
N = 1000                        # Number of samples

# Load images and pre-process:
images = Images(test, selection_category, N, image_size)
X_train, X_val, y_train, y_val = images.load_images(path_to_images)

# Save the data
#np.save("{}X_train_{}_{}".format(path_to_save, selection_category, test), X_train, allow_pickle=False)
#np.save("{}y_train_{}_{}".format(path_to_save, selection_category, test), y_train, allow_pickle=False)
#np.save("{}X_val_{}_{}".format(path_to_save, selection_category, test), X_val, allow_pickle=False)
#np.save("{}y_val_{}_{}".format(path_to_save, selection_category, test), y_val, allow_pickle=False)

# CNN
epoch = 50
batch_size = 64

#Load tuned and compiled model 
mymodel = LoadCNN(test)

print(mymodel.model.summary())

# Fit model 
mymodel.fit_model(X_train, y_train, batch_size, epoch, aug = True, validation_data = (X_val, y_val),
                  file_model = "{}CNN_model_{}_{}.h5".format(path_to_save, selection_category, test),
                  file_hist = "{}CNN_hist__{}_{}.txt".format(path_to_save, selection_category, test))

# Plot loss & accuracy plot and confusion matrix
mymodel.vis_acc_loss("{}CNN_lossAcc_{}_{}.eps".format(path_to_figs, selection_category, test))
mymodel.vis_cm(X_val, y_val,
               file_fig = "{}CNN_cm_{}_{}.eps".format(path_to_figs, selection_category, test),
               file_mat = "{}CNN_cm_{}_{}.txt".format(path_to_save, selection_category, test))

