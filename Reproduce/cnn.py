#!/usr/bin/env python3
"""
Create Convolutional Neural Network(CNN) model and train with images

@author: ulas isildak
"""

import sys
sys.path.insert(0, '/Users/ulas/Projects/balancing_selection')
from BaSe.Preprocess import Images
from BaSe.Model import LoadCNN

path_to_image = '/Users/ulas/Projects/balancing_selection/Data/Images/'
path_to_save = '/Users/ulas/Projects/balancing_selection/Data/Model/'
path_to_figs = '/Users/ulas/Projects/balancing_selection/Results/Figures/'

N = 20000
img_dim = (128, 128)

for test in [1, 2, 3]:
    for selection_category in ["recent", "medium", "old"]:
        images = Images(test, selection_category, N, img_dim)
        X_train, X_val, y_train, y_val = images.load_images(path_to_image, val_size=0.2, toshuffle=True,
                                                            random_state=2, verbose=1)

        mymodel = LoadCNN(test, img_dim)
        mymodel.summary()

        mymodel.fit_model(X_train, y_train, batch_size=32, epoch=10, aug=True, validation_data=(X_val, y_val), verbose=2)

        mymodel.save_model("{}CNN_model_{}_{}.h5".format(path_to_save, selection_category, test))

        mymodel.vis_acc_loss(file="{}CNN_lossAcc_{}_{}.eps".format(path_to_figs, selection_category, test))
        mymodel.vis_cm(X_val, y_val, images.classes,
                       file_fig="{}CNN_cm_{}_{}.eps".format(path_to_figs, selection_category, test))
