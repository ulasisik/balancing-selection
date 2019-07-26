#!/usr/bin/env python3
"""
Visualizes decisions for trained CNN

@author: ulas isildak
@e-mail: isildak.ulas [at] gmail.com
"""

import numpy as np
import scipy.ndimage as ndimage

import matplotlib
matplotlib.use('Agg')   #or use ssh -X to connect to sever
import matplotlib.pyplot as plt
from matplotlib import cm
#tensorflow v.1.10
from keras import activations
from keras import backend as K
from keras.models import load_model
#keras-vis v.0.4.1
from vis.utils import utils 
from vis.visualization import visualize_saliency
from vis.visualization import visualize_cam

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last') #for Theano
print("Image_data_format is " + K.image_data_format())


class Vis(object):
    
    figsize=(6,6)
    
    def __init__(self, model, X, y, test, image_size=(128,128), input_class=None):
        '''
        Loads trained CNN model and data for visualization, and selects an input 
        image to use in visualization
        
        Args:
            model: Full path to trained CNN model(h5 object)
            X: Full path to test set of images (npy object)
            y: Full path to class labels of x  (npy object)
            test: Test number; either 1, 2, or 3:
                1 - Neutrality vs selection
                2 - Incomplete sweep vs balancing selection
                3 - Overdominance vs negative freq-dependent selection
            image_size: Image dimensions as (rows, columns). Default is (128, 128)
            input_class: Class of the selected image. If None (default), returns a random selection
        '''
        self.model = load_model(model)
        self.X = np.load(X)
        self.y = np.load(y)
        self.test = test
        if test == 1:
            self.mapings = {0: "Neutral", 1: "Selection"}
        elif test == 2:
            self.mapings = {0: "Incomplete Sweep", 1: "Balancing Selection"}
        elif test ==3:
            self.mapings = {0: "Overdominance", 1: "Negative Freq-Dep. Selection"}
        else: 
            raise ValueError("Invalid test number {}".format(test))
        self.classes = list(map(self.mapings.get, self.y.tolist()))
        self.image_size=image_size
        self.image_rows=image_size[0]
        self.image_cols=image_size[1]
        if input_class in self.classes:
            idx = np.random.choice(np.where(np.array(self.classes) == input_class)[0],1)[0]
            self.input_image = self.X[idx,:,:,0]
            self.input_class = input_class
        elif input_class == None:
            idx = np.random.choice(range(len(self.classes)), 1)[0]
            self.input_image = self.X[idx,:,:0]
            self.input_class = self.classes[idx]
        else:
            raise IndexError("Class '{}' is not testted in test {}: {} vs {}." \
                             .format(input_class, self.test, list(self.mapings.values())[0], list(self.mapings.values())[1]))
        self.pred_class_input = self.mapings[int(self.model.predict(self.input_image.reshape(1,self.image_rows,self.image_cols,1))[0][0]>0.5)]
        
        if [lb for lb, cl in self.mapings.items() if cl == self.input_class][0] == 0:
            self.pred_prob_input = 1 - self.model.predict(self.input_image.reshape(1,self.image_rows,self.image_cols,1))[0][0]
        elif [lb for lb, cl in self.mapings.items() if cl == self.input_class][0] == 1:
            self.pred_prob_input = self.model.predict(self.input_image.reshape(1,self.image_rows,self.image_cols,1))[0][0]
        
        
    def vis_input(self, file):
        '''
        Visualizes input image
        
        Args:
            file: file name of the image, including full path
        '''
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(self.input_image, cmap=cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.title('Input Image\nTrue class: {0}\n Predicted class: {1} (probability:{2:.2f})'.format(self.input_class, self.pred_class_input, self.pred_prob_input))
        plt.savefig(file)
        
           
    def vis_raw_weights(self, file, layer_idx=1, filter_idx=None):
        '''
        Visualizes raw weights(kernels) for given convolutional layer.
        
        Args:
            file: file name of the image, including full path
            conv_layer: Index of the convolutional layer to visualize. Default is the first conv layer.
            filter_idx: Index of the filter to visualize. Default is 'None' for all filters.
        '''
        clayers = [layer for layer in self.model.layers if 'convolutional' in str(layer)]
        if layer_idx > len(clayers):
            raise IndexError("Convolutional layer index {} out of range.".format(layer_idx))
        clayer = clayers[layer_idx-1]
        kernels=clayer.get_weights()[0]
        kernels=kernels.reshape(kernels.shape[3], kernels.shape[0], kernels.shape[1]) #assumed to have shape of (kernel_sizes,channels,filters)
        if filter_idx == None:
            fig = plt.figure()
            fig.suptitle('Kernels of Conv_{} Layer'.format(layer_idx), fontsize=14)
            for j in range(len(kernels)):
                ax = fig.add_subplot((kernels.shape[0]//8 + (kernels.shape[0]%8 >0)),8,j+1)
                ax.matshow(kernels[j], cmap = matplotlib.cm.binary)
                plt.title(str(j+1),fontsize=8)
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
                fig.subplots_adjust(top=0.88)
        elif filter_idx <= len(kernels):
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(1,1,1)
            ax.matshow(kernels[filter_idx-1], cmap = matplotlib.cm.binary)
            plt.title('{}. Filter  of Conv_{} Layer'.format(filter_idx, layer_idx))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.close()
        fig.savefig(file)
    
    
    def vis_activations(self, file, layer_idx=1, filter_idx=None):
        '''
        Visualizes the result of activation layers by applying them on an input image
        
        Args:
            file: file name of the image, including full path
            layer_idx: Index of the activation layer to visualize. Default is 
                the first activation layer.
            filter_idx: Index of the filter to visualize. Default is 'None' for all filters.
        '''
        act_layers = [layer for layer in self.model.layers if 'Activation' in str(layer)]
        if layer_idx > len(act_layers):
            raise IndexError("Activation layer index {} out of range.".format(layer_idx))
        input_layer = self.model.layers[0]
        output_layer = act_layers[layer_idx]
        
        output_fn = K.function([input_layer.input, K.learning_phase()],[output_layer.output])
        output_image = output_fn([self.input_image.reshape(1, self.image_rows, self.image_cols, 1),1])[0]
        
        if filter_idx == None:
            fig = plt.figure()
            fig.suptitle('Results of Activation_{}\n(input image class: "{}")'\
                         .format(layer_idx, self.input_class), fontsize=14)
            for j in range(output_image.shape[-1]):
                ax = fig.add_subplot((output_image.shape[-1]//8 + (output_image.shape[-1]%8 >0)),8,j+1)
                ax.imshow(output_image[0,:,:,j],cmap=plt.cm.hot)
                plt.title(str(j+1),fontsize=8)
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
                fig.subplots_adjust(top=0.82)
        elif filter_idx <= output_image.shape[-1]:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(1,1,1)
            ax.imshow(output_image[0,:,:,filter_idx-1],cmap=plt.cm.hot)
            plt.title('Result of {}. Filter for Activation_{}\n(input image class: "{}")'\
                      .format(filter_idx, layer_idx, self.input_class))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.savefig(file)
    
    
    @staticmethod
    def iter_occlusion(image, occlusion_size=8):
        '''
        Iterate occlusion over regions of the input image.
        
        Args:
            image: (np array) Input image in shape (img_row, img_col, channel)
            occlusion_size: (int) Size of the occlusion. Default is 8x8
        
        Adapted from https://www.kaggle.com/blargl/simple-occlusion-and-saliency-maps
        '''
        occlusion = np.full((occlusion_size * 5, occlusion_size * 5, 1), [0.5], np.float32)
        occlusion_center = np.full((occlusion_size, occlusion_size, 1), [0.5], np.float32)
        occlusion_padding = occlusion_size * 2
        
        print('Padding...')
        image_padded = np.pad(image, ((occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0)), 'constant', constant_values = 0.0)
        
        for y in range(occlusion_padding, image.shape[0] + occlusion_padding, occlusion_size):
        
            for x in range(occlusion_padding, image.shape[1] + occlusion_padding, occlusion_size):
                tmp = image_padded.copy()
                
                tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding, x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] = occlusion
                
                tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center
                
                yield x - occlusion_padding, y - occlusion_padding, \
                tmp[occlusion_padding:tmp.shape[0] - occlusion_padding, occlusion_padding:tmp.shape[1] - occlusion_padding]
    
    
    def vis_occlusion(self, file, occlusion_size=8):
        '''
        Visualizes occlusion heatmap.
        
        Args:
            file: file name of the image, including full path
            occlusion_size: (intoger speficying occlusion size. Default is 8x8
        '''
        
        image = self.input_image.reshape(self.image_rows, self.image_cols, 1)
        inlab = [lb for lb, cl in self.mapings.items() if cl == self.input_class][0]
        
        heatmap = np.zeros(self.image_size, np.float32)

        for n, (x, y, img_float) in enumerate(self.iter_occlusion(image, occlusion_size=occlusion_size)):
            X = img_float.reshape(1, 128, 128, 1)
            out = self.model.predict(X)
            if inlab:
                print('#{0}: {1} @ {2: .4f} (correct class: {3})'.format(n, self.mapings[int(out>0.5)], np.max(out), self.input_class))
                heatmap[y:y + occlusion_size, x:x + occlusion_size] = out[0][0]
            elif not inlab:
                print('#{0}: {1} @ {2: .4f} (correct class: {3})'.format(n, self.mapings[int(out>0.5)], (1-np.max(out)), self.input_class))
                heatmap[y:y + occlusion_size, x:x + occlusion_size] = 1-out[0][0]
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(self.input_image, cmap=cm.gray)
        im = ax.pcolormesh(heatmap, cmap=plt.cm.jet, alpha=0.40)
        plt.colorbar(im,fraction=0.046, pad=0.04).solids.set(alpha=1)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.title('Occlusion Heatmap\nTrue: {0}, Predicted: {1}'.format(self.input_class, self.pred_class_input), fontsize=16)
        plt.savefig(file)

    
    def vis_pca(self, file, n=None):
        '''
        Similar to vis_tsne, visualizes feature vectors of last fully conneted 
        layer (the layer immediately before the classifier) obtained  by runing 
        the network on a set of input images by PCA
        
        Args:
            file: file name of the image, including full path
            n: Number of samples used. If None (default), use all samples
        '''
        output_fn = K.function([self.model.layers[0].input, K.learning_phase()],[self.model.layers[-4].output])
        vis_dense= np.zeros(shape=(1,int(self.model.layers[-2].output.shape[1])))
        if n:
            n_samp = n
        else:
            n_samp = self.y.shape[0]
        
        for i in range(n_samp):
            input_image = self.X[i,:,:,:].reshape((1,128,128,1))
            output = output_fn([input_image,1])[0]
            vis_dense=np.concatenate((vis_dense, output), axis=0)
        X = vis_dense[1:,:]
        y = np.array(self.classes[:n_samp])
        
        pca = PCA(n_components=2)
        pca.fit(X)
        pca_result = pca.transform(X)
            
        per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)
        ax.scatter(pca_result[y == list(self.mapings.values())[0],0], pca_result[y == list(self.mapings.values())[0],1], label=list(self.mapings.values())[0], color='dodgerblue')
        ax.scatter(pca_result[y == list(self.mapings.values())[1],0], pca_result[y == list(self.mapings.values())[1],1], label=list(self.mapings.values())[1], color='orangered')
        ax.legend()
        plt.title('PCA', fontsize=16)
        plt.xlabel('PC1 ({}%)'.format(per_var[0]), fontsize=12)
        plt.ylabel('PC2 ({}%)'.format(per_var[1]), fontsize=12)
        plt.savefig(file)
    
    
    def vis_tsne(self, file, n=None):
        '''
        Similar to vis_pca, visualizes feature vectors of last fully conneted layer 
        (the layer immediately before the classifier) obtained by runing the network 
        on a set of input images by t-SNE.
        
        Args:
            file: file name of the image, including full path
            n: Number of samples used. If None (default), use all samples
        '''
        
        output_fn = K.function([self.model.layers[0].input, K.learning_phase()],[self.model.layers[-4].output])
        vis_dense= np.zeros(shape=(1,int(self.model.layers[-2].output.shape[1])))
        if n:
            n_samp = n
        else:
            n_samp = self.y.shape[0]
        
        for i in range(n_samp):
            input_image = self.X[i,:,:,:].reshape((1,128,128,1))
            output = output_fn([input_image,1])[0]
            vis_dense=np.concatenate((vis_dense, output), axis=0)
        X = vis_dense[1:,:]
        y = np.array(self.classes[:n_samp])
        
        tsne = TSNE(n_components=2, verbose=0)
        tsne_result = tsne.fit_transform(X)
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)
        ax.scatter(tsne_result[y == list(self.mapings.values())[0],0], tsne_result[y == list(self.mapings.values())[0],1], label=list(self.mapings.values())[0], color='dodgerblue')
        ax.scatter(tsne_result[y == list(self.mapings.values())[1],0], tsne_result[y == list(self.mapings.values())[1],1], label=list(self.mapings.values())[1], color='orangered')
        ax.legend()
        plt.title('t-SNE', fontsize=16)
        plt.xlabel('')
        plt.ylabel('')
        plt.savefig(file)
    
    
    def vis_saliency(self, file):
        '''
        Generates an attention heatmap over the input_image, highlighting the 
        salient image regions that contribute the most towards the output        
        https://arxiv.org/pdf/1312.6034v2.pdf.
        
        Args:
            file: file name of the image, including full path
        '''
        layer_idx = -1  #last layer
        
        inp = self.input_image.reshape(self.image_rows, self.image_cols, 1)
        model = self.model
        # Swap sigmoid with linear
        model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(model)
        
        grads=visualize_saliency(model, layer_idx=layer_idx, filter_indices=[0], seed_input = inp)
        smoothe = ndimage.gaussian_filter(grads, sigma=2) 
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(self.input_image, cmap=cm.gray)
        ax.imshow(smoothe, alpha=0.70, cmap=plt.cm.seismic)
        plt.title('Saliency Map\nTrue: {0}, Predicted: {1}'.format(self.input_class, self.pred_class_input), fontsize=16)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.savefig(file)
        
    
    def vis_gradcam(self, file):
        '''
        Generates a gradient based class activation map (grad-CAM) that maximizes 
        the output of final keras Dense layer.
        https://arxiv.org/pdf/1610.02391.pdf
        
        Args:
            file: file name of the image, including full path
        '''
        layer_idx = -1
        
        inp = self.input_image.reshape(self.image_rows, self.image_cols, 1)
        model = self.model
        # Swap sigmoid with linear
        model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(model)
        
        cam_grads = visualize_cam(model, layer_idx, filter_indices=[0], seed_input=inp)
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(self.input_image, cmap=cm.gray)
        ax.imshow(cam_grads, alpha=0.70, cmap=plt.cm.jet)
        plt.title('Grad-CAM\nTrue: {0}, Predicted: {1}'.format(self.input_class, self.pred_class_input), fontsize=16)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.savefig(file)

