#!/usr/bin/env python3
"""
Perform prediction on genomic data using trained models

@author: ulas isildak
"""

import os
import sys
import numpy as np
sys.path.insert(0, '/Users/ulas/Projects/balancing_selection')

from BaSe.Preprocess import VCF
from BaSe.Model import predict

file_name = "/Users/ulas/Projects/balancing_selection/Data/VCFs/test.vcf"

vcf = VCF(file_name)

im_matrix, snps, pos = vcf.create_image(N=50000, sort="freq", method="s", target_freq=(0.4, 0.6),
                                        target_list=None, target_range=None, img_dim=(128, 128))

stat_matrix, _, _ = vcf.create_stat(N=50000, target_freq=(0.4, 0.6), target_list=None,
                                    target_range=None, scale=True, pca=False)

# Save stats, images and variant ids
# np.save("{}test_stat.npy".format(save), stat_matrix)
# np.save("{}test_im.npy".format(save), im_matrix)
# np.save("{}test_snpid.npy".format(save), snps)

# Load stats, images and variant ids
# load = "/Users/ulas/Projects/balancing_selection/Data/VCFs/"
# stat_matrix = np.load("{}mefv_stat_filter01.npy".format(load))
# im_matrix = np.load("{}mefv_im_filter01.npy".format(load))
# snps = np.load("{}mefv_snpid_filter01.npy".format(load))
# pos = np.load("{}mefv_pos_filter01.npy".format(load))

save = "/Users/ulas/Projects/balancing_selection/Results/Predictions/"
modelsdir = "/Users/ulas/Projects/balancing_selection/Data/Model"
models = [m for m in os.scandir(modelsdir) if m.name.endswith("h5")]

for model in models:
    print(model.name)
    fname = model.name.replace(".h5", "").replace("model", "mefv")
    if fname.startswith("ANN"):
        pass
    elif fname.startswith("CNN"):
        predict(im_matrix, model.path, snps, pos, int(fname.split("_")[-1]), file="{}{}.csv".format(save, fname))
    else:
        raise ValueError("{} is not a valid keras model".format(fname))
