#!/usr/bin/env python3
"""
Visualizes CNN results

@author: ulas isildak
@e-mail: isildak.ulas [at] gmail.com
"""
import random
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from vis.utils import utils
from vis.visualization import visualize_cam

from keras import activations
from keras import backend as K
from keras.models import load_model
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_last')

class Vis(object):

    def __init__(self, model, x, y, test, img_dim=(128,128)):
        """
        Loads trained CNN model and data for visualization

        Parameters:
            model: Full path to trained CNN model(h5 object)
            x: Full path to test set of images (npy object)
            y: Full path to class labels of x  (npy object)
            test: Test number; either 1, 2, or 3:
                1 - Neutrality vs selection
                2 - Incomplete sweep vs balancing selection
                3 - Overdominance vs negative freq-dependent selection
            img_dim: Image dimensions as (rows, columns). Default is (128, 128)
        """
        if test == 1:
            self.mapings = {0: "Neutral", 1: "Selection"}
        elif test == 2:
            self.mapings = {0: "Incomplete Sweep", 1: "Balancing Selection"}
        elif test == 3:
            self.mapings = {0: "Overdominance", 1: "Negative Freq-Dep. Selection"}
        else:
            raise ValueError("Invalid test number {}".format(test))

        self.test = test
        self.model = load_model(model)
        self.x = np.load(x)
        self.y = np.load(y)

        self.classes = list(map(self.mapings.get, self.y.tolist()))
        self.image_rows = img_dim[0]
        self.image_cols = img_dim[1]

    def select_input(self, input_class=None, seed=None):
        """
        Randomly selects an input image to use in visualization

        Paramters:
            input_class: Class of the selected image. If None (default), returns a random selection
            seed: Random seed

        """
        if not input_class:
            input_class = random.choices(self.classes)[0]
        elif input_class not in self.mapings.values():
            raise ValueError("Invalid input class name: {}".format(input_class))

        random.seed(seed)
        idx = random.choices(np.where(np.array(self.classes) == input_class)[0])[0]
        self.input_image = self.x[idx, :, :, 0]
        self.input_class = input_class

        self.pred_value = self.model.predict(self.input_image.reshape(1, self.image_rows, self.image_cols, 1))[0][0]
        self.pred_class = self.mapings[int(self.pred_value > 0.5)]
        self.pred_prob = abs(self.pred_value - 0.5)/0.5

    def vis_input(self, file=None, figsize=(6, 6)):
        """Visualizes input image

            Parameterss:
                file: File name of the figure to be saved
                figsize: Figure size (row, col)
            '''
        """

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.input_image, cmap=cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.title('Input Image\nTrue class: {0}\n Predicted class: {1} (probability:{2:.2f})'.format(self.input_class,
                                                                                                     self.pred_class,
                                                                                                     self.pred_prob))
        if file:
            plt.savefig(file)
        plt.show()

    def vis_raw_weights(self, file=None, layer_idx=1, filter_idx=None, figsize=(6, 6)):
        """
        Visualizes raw weights(kernels) for given convolutional layer.

        Parameters:
            file: File name of the figure to be saved
            layer_idx: Index of the convolutional layer to visualize. Default is the first conv layer.
            filter_idx: Index of the filter to visualize. Default is 'None' for all filters.
            figsize: Figure size (height, width)
        """

        clayers = [layer for layer in self.model.layers if 'convolutional' in str(layer)]
        if layer_idx > len(clayers):
            raise IndexError("Convolutional layer index {} out of range.".format(layer_idx))
        clayer = clayers[layer_idx - 1]
        kernels = clayer.get_weights()[0]
        # assumes to have shape of (kernel_sizes,channels,filters)
        kernels = kernels.reshape(kernels.shape[3], kernels.shape[0], kernels.shape[1])

        if not filter_idx:
            fig = plt.figure(figsize=figsize)
            fig.suptitle('Kernels of Conv_{} Layer'.format(layer_idx), fontsize=14)
            for j in range(len(kernels)):
                ax = fig.add_subplot((kernels.shape[0] // 8 + (kernels.shape[0] % 8 > 0)), 8, j + 1)
                ax.matshow(kernels[j], cmap=cm.binary)
                plt.title(str(j + 1), fontsize=8)
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
                fig.subplots_adjust(top=0.88)
                plt.show()
        elif filter_idx <= len(kernels):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
            ax.matshow(kernels[filter_idx - 1], cmap=cm.binary)
            plt.title('{}. Filter  of Conv_{} Layer'.format(filter_idx, layer_idx))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.show()
        else:
            raise IndexError("Filter index {} out of range.".format(filter_idx))
        if file:
            fig.savefig(file)

    def vis_activations(self, file=None, layer_idx=1, filter_idx=None, figsize=(6, 6)):
        """
        Applies activations on the input image and visualizes the result

        Parameters:
            file: File name of the figure to be saved
            layer_idx: Index of the activation layer to visualize. Default is the first activation layer.
            filter_idx: Index of the filter to visualize. Default is 'None' for all filters.
            figsize: Figure size (height, width)
        """

        act_layers = [layer for layer in self.model.layers if 'Activation' in str(layer)]
        if layer_idx > len(act_layers):
            raise IndexError("Activation layer index {} out of range.".format(layer_idx))
        input_layer = self.model.layers[0]
        output_layer = act_layers[layer_idx - 1]

        output_fn = K.function([input_layer.input, K.learning_phase()], [output_layer.output])
        output_image = output_fn([self.input_image.reshape(1, self.image_rows, self.image_cols, 1), 1])[0]

        if not filter_idx:
            fig = plt.figure(figsize=figsize)
            fig.suptitle('Results of Activation_{}\n(input image class: "{}")'
                         .format(layer_idx, self.input_class), fontsize=14)
            for j in range(output_image.shape[-1]):
                ax = fig.add_subplot((output_image.shape[-1] // 8 + (output_image.shape[-1] % 8 > 0)), 8, j + 1)
                ax.imshow(output_image[0, :, :, j], cmap=plt.cm.hot)
                plt.title(str(j + 1), fontsize=8)
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.tight_layout()
                fig.subplots_adjust(top=0.82)
                plt.show()
        elif filter_idx <= output_image.shape[-1]:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(output_image[0, :, :, filter_idx - 1], cmap=plt.cm.hot)
            plt.title('Result of {}. Filter for Activation_{}\n(input image class: "{}")'
                      .format(filter_idx, layer_idx, self.input_class))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.show()
        else:
            raise IndexError("Filter index {} out of range.".format(filter_idx))
        if file:
            plt.savefig(file)

    def vis_pca(self, file=None, n=None, figsize=(6, 6)):
        """
        Visualizes feature vectors of last fully conneted layer (the layer immediately before the classifier)
        obtained  by runing the network on a set of input images by PCA.

        Parameters:
            file: File name of the figure to be saved
            n: Number of samples used. If None (default), use all samples
            figsize: Figure size (height, width)

        """
        output_layer = [layer for layer in self.model.layers if 'Dense' in str(layer)][-2]
        output_fn = K.function([self.model.layers[0].input, K.learning_phase()], [output_layer.output])
        vis_dense = np.zeros(shape=(1, int(output_layer.output.shape[1])))

        if not n:
            n = self.y.shape[0]

        for i in range(n):
            input_image = self.x[i, :, :, :].reshape((1, 128, 128, 1))
            output = output_fn([input_image, 1])[0]
            vis_dense = np.concatenate((vis_dense, output), axis=0)
        X = vis_dense[1:, :]
        y = np.array(self.classes[:n])

        pca = PCA(n_components=2)
        pca.fit(X)
        pca_result = pca.transform(X)

        per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(pca_result[y == list(self.mapings.values())[0], 0],
                   pca_result[y == list(self.mapings.values())[0], 1],
                   label=list(self.mapings.values())[0], color='dodgerblue')
        ax.scatter(pca_result[y == list(self.mapings.values())[1], 0],
                   pca_result[y == list(self.mapings.values())[1], 1],
                   label=list(self.mapings.values())[1], color='orangered')
        ax.legend()
        plt.title('PCA', fontsize=16)
        plt.xlabel('PC1 ({}%)'.format(per_var[0]), fontsize=12)
        plt.ylabel('PC2 ({}%)'.format(per_var[1]), fontsize=12)
        plt.show()
        if file:
            plt.savefig(file)

    def vis_tsne(self, file=None, n=None, figsize=(6, 6)):
        """
        Visualizes feature vectors of last fully conneted layer (the layer immediately before the classifier)
        obtained  by runing the network on a set of input images by t-SNE.

        Parameters:
            file: File name of the figure to be saved
            n: Number of samples used. If None (default), use all samples
            figsize: Figure size (height, width)

        """
        output_layer = [layer for layer in self.model.layers if 'Dense' in str(layer)][-2]
        output_fn = K.function([self.model.layers[0].input, K.learning_phase()], [output_layer.output])
        vis_dense = np.zeros(shape=(1, int(output_layer.output.shape[1])))

        if not n:
            n = self.y.shape[0]

        for i in range(n):
            input_image = self.x[i, :, :, :].reshape((1, 128, 128, 1))
            output = output_fn([input_image, 1])[0]
            vis_dense = np.concatenate((vis_dense, output), axis=0)
        X = vis_dense[1:, :]
        y = np.array(self.classes[:n])

        tsne = TSNE(n_components=2, verbose=0)
        tsne_result = tsne.fit_transform(X)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(tsne_result[y == list(self.mapings.values())[0], 0],
                   tsne_result[y == list(self.mapings.values())[0], 1],
                   label=list(self.mapings.values())[0], color='dodgerblue')
        ax.scatter(tsne_result[y == list(self.mapings.values())[1], 0],
                   tsne_result[y == list(self.mapings.values())[1], 1],
                   label=list(self.mapings.values())[1], color='orangered')
        ax.legend()
        plt.title('t-SNE', fontsize=16)
        plt.xlabel('')
        plt.ylabel('')
        plt.show()
        if file:
            plt.savefig(file)

    @staticmethod
    def iter_occlusion(image, occlusion_size):
        """
        Iterate occlusion over regions of the input image.

        Parameters:
            image: Input image in shape (img_row, img_col, channel)
            occlusion_size: Size of the occlusion.

        Adapted from https://www.kaggle.com/blargl/simple-occlusion-and-saliency-maps
        """
        occlusion = np.full((occlusion_size * 5, occlusion_size * 5, 1), [0.5], np.float32)
        occlusion_center = np.full((occlusion_size, occlusion_size, 1), [0.5], np.float32)
        occlusion_padding = occlusion_size * 2

        print('Padding...')
        image_padded = np.pad(image,
                              ((occlusion_padding, occlusion_padding), (occlusion_padding, occlusion_padding), (0, 0)),
                              'constant', constant_values=0.0)

        for y in range(occlusion_padding, image.shape[0] + occlusion_padding, occlusion_size):

            for x in range(occlusion_padding, image.shape[1] + occlusion_padding, occlusion_size):
                tmp = image_padded.copy()

                tmp[y - occlusion_padding:y + occlusion_center.shape[0] + occlusion_padding,
                x - occlusion_padding:x + occlusion_center.shape[1] + occlusion_padding] = occlusion

                tmp[y:y + occlusion_center.shape[0], x:x + occlusion_center.shape[1]] = occlusion_center

                yield x - occlusion_padding, y - occlusion_padding, \
                      tmp[occlusion_padding:tmp.shape[0] - occlusion_padding,
                      occlusion_padding:tmp.shape[1] - occlusion_padding]

    def vis_occlusion(self, file=None, occlusion_size=8, figsize=(6, 6), verbose=0):
        """
        Visualizes occlusion heatmap.

        Parameters:
            file: File name of the figure to be saved
            occlusion_size: an int speficying occlusion size. Default is 8x8
            figsize: Figure size (height, width)
            verbose: Specifies verbosity
        """
        image = self.input_image.reshape(self.image_rows, self.image_cols, 1)
        inlab = list(self.mapings.keys())[list(self.mapings.values()).index(self.input_class)]

        heatmap = np.zeros((self.image_rows, self.image_cols), np.float32)
        for n, (x, y, img_float) in enumerate(self.iter_occlusion(image, occlusion_size=occlusion_size)):
            X = img_float.reshape(1, 128, 128, 1)
            out = self.model.predict(X)
            if inlab:
                heatmap[y:y + occlusion_size, x:x + occlusion_size] = abs(out[0][0] - 0.5) / 0.5
                if verbose > 0:
                    print('#{0}: {1} @ {2: .4f} (correct class: {3})'.format(n, self.mapings[int(out > 0.5)],
                                                                             abs(out[0][0] - 0.5) / 0.5,
                                                                             self.input_class))

            else:
                heatmap[y:y + occlusion_size, x:x + occlusion_size] = abs(out[0][0] - 0.5) / 0.5
                if verbose > 0:
                    print('#{0}: {1} @ {2: .4f} (correct class: {3})'.format(n, self.mapings[int(out > 0.5)],
                                                                             abs(out[0][0] - 0.5) / 0.5,
                                                                             self.input_class))

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.input_image, cmap=cm.gray)
        im = ax.pcolormesh(heatmap, cmap=plt.cm.jet, alpha=0.40)
        plt.colorbar(im, fraction=0.046, pad=0.04).solids.set(alpha=1)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.title('Occlusion Heatmap\nTrue: {0}, Predicted: {1}'.format(self.input_class, self.pred_class), fontsize=16)
        plt.show()
        if file:
            plt.savefig(file)

    def vis_gradcam(self, file=None, figsize=(6, 6)):
        """
        Generates a gradient based class activation map (grad-CAM) that maximizes
        the output of final keras Dense layer. https://arxiv.org/pdf/1610.02391.pdf

        Parameters:
            file: File name of the figure to be saved
            figsize: Figure size (height, width)
        """
        layer_idx = -1
        inlab = list(self.mapings.keys())[list(self.mapings.values()).index(self.input_class)]

        if inlab == 0:
            grad_mod = "invert"
        else:
            grad_mod = None

        inp = self.input_image.reshape(self.image_rows, self.image_cols, 1)
        model = self.model
        # Swap sigmoid with linear
        model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(model)

        cam_grads = visualize_cam(model, layer_idx, filter_indices=[0], seed_input=inp, grad_modifier=grad_mod)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.input_image, cmap=cm.gray)
        ax.imshow(cam_grads, alpha=0.70, cmap=plt.cm.jet)
        plt.title('Grad-CAM\nTrue: {0}, Predicted: {1}'.format(self.input_class, self.pred_class), fontsize=16)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.show()
        if file:
            plt.savefig(file)
