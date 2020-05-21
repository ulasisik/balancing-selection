#!/usr/bin/env python3
"""
Visualizes CNN results

@author: ulas isildak
@e-mail: isildak.ulas [at] gmail.com
"""
import sys
sys.path.insert(0, '/Users/ulas/Projects/balancing_selection')

from BaSe import vis_cnn

X = "/mnt/NEOGENE1/projects/deepLearn_selection_2018/cnn/model/X_test.npy"
y = "/mnt/NEOGENE1/projects/deepLearn_selection_2018/cnn/model/y_test.npy"
model = "/Users/ulas/Projects/balancing_selection/Data/Model/CNN_model_recent_1.h5"
path_to_figs = '/mnt/NEOGENE1/projects/deepLearn_selection_2018/cnn/figs/vis/'

test = 1
image_size = (128,128)
input_class = 'Neutral'

#Load test data and model; select Neutral class to visualize
vis=vis_cnn.Vis(model, X, y, test, image_size, input_class=input_class)

#Input Image
vis.vis_input(path_to_figs+'input.png')

#raw weights(kernels)
vis.vis_raw_weights(path_to_figs+'raw_weights.png', layer_idx=1)

#activations(image through filters of conv layer)
vis.vis_activations(path_to_figs+'activations.png', layer_idx=1)

#pca and t-sne of feature vectors of last layer
vis.vis_pca(path_to_figs+'pca.png', n=100)
vis.vis_tsne(path_to_figs+'tsne.png', n=100)

#occlusion
vis.vis_occlusion(path_to_figs+'occlusion.png', occlusion_size=8)

#saliency map
vis.vis_saliency(path_to_figs+'saliency.png')

#grad-cam
vis.vis_gradcam(path_to_figs+'gradcam.png')

