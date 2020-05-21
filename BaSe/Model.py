#!/usr/bin/env python3
"""
Contains required modules to create, train and evaluate deep neural network based classifier

@author: ulas isildak
@e-mail: isildak.ulas [at] gmail.com
"""

import numpy as np
import pandas as pd
from math import ceil
from itertools import chain
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from keras import optimizers
from keras import regularizers
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D 
from keras.layers import Flatten 
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

class MakeModel:
    """Abstract class for creating models and visualizing result"""

    def __init__(self, model):
        self.model = model
        self.epoch = None
        self.train_loss = None
        self.val_loss = None
        self.train_acc = None
        self.val_acc = None

    @staticmethod
    def make_dense_block(input_tensor, unit, l2_regularizer, initializer, dropout, batch=True):
        """
        Creates a hidden layer
        
        Parameters:
            input_tensor: input tensor
            unit: Number of units
            l2_regularizer: l2 weight decay
            initializer: kernel initializer to be used in the dense layer
            unit: number of neurons(units) in the dense layer
            dropout: dropout rate
            batch: if True, adds batch normalization
        Returns:
            Keras dense layer
        """
        
        x = Dense(unit, kernel_initializer=initializer,
                  kernel_regularizer=regularizers.l2(l2_regularizer))(input_tensor)
        if batch:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout:
            x = Dropout(dropout)(x)
        return x

    @staticmethod
    def make_conv_block(input_tensor, filters, kernel_size, pooling_size, l2_regularizer, initializer,
                        strides, padding, batch=True, dropout=None, pooling=True):
        """
        Creates a convolutional block
        
        Parameters:
            input_tensor: input tensor
            filters: number of filter
            kernel_size: kernel size (kernel_size, kernel_size)
            pooling_size: the size of maximum pooling
            l2_regularizer: l2 weight decay
            initializer: kernel initializer
            strides: a tuple of 2 integers specifying the strides along the height and width
            padding: one of "valid" or "same"
            batch: if True, performs batch normalization
            dropout: dropout rate
            pooling: if True performs MaxPooling2D
        Returns:
            Keras convolıtional layer
        """
        
        x = Conv2D(filters, (kernel_size, kernel_size), strides=strides, padding=padding,
                   data_format='channels_last', kernel_regularizer=regularizers.l2(l2_regularizer),
                   kernel_initializer=initializer)(input_tensor)
        if batch:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if pooling:
            x = MaxPooling2D(pool_size=(pooling_size, pooling_size))(x)
        if dropout:
            x = Dropout(dropout)(x)
        return x

    def compile_model(self, lr, lr_dec):
        """
        Compiles cnn model using Adam optimizer.
        
        Parameters:
            lr: learning rate
            lr_dec: learning rate decay
        """
        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                           optimizer=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=lr_dec))

    def summary(self):
        """Return model summary"""
        return self.model.summary()
        
    def fit_model(self, x, y, batch_size, epoch, aug=False, validation_data=None, verbose=0):
        """
        Fits model to the data
        
        Parameters:
            x: training set
            y: corresponding labels
            batch_size: batch size
            epoch: epoch
            aug: if True, performs image agumentation. Default is False.
            validation_data: validation data (X_val, y_val)
            verbose: specifies verbosity mode:
                0: silent,
                1: progress bar,
                2: one line per epoc
        """
        self.epoch = epoch

        if aug:
            datagen_train = ImageDataGenerator(
                samplewise_center=False,  # set each sample mean to 0
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images
            datagen_train.fit(x)

            steps = ceil(x.shape[0]/batch_size)

            if validation_data:
                hist = self.model.fit_generator(datagen_train.flow(x, y, batch_size=batch_size), epochs=epoch,
                                                validation_data=validation_data, steps_per_epoch=steps, verbose=verbose)
            else:
                hist = self.model.fit_generator(datagen_train.flow(x, y, batch_size=batch_size),
                                                epochs=epoch, steps_per_epoch=steps, verbose=verbose)

        else:
            if validation_data:
                hist = self.model.fit(x, y, batch_size=batch_size, epochs=epoch,
                                      validation_data=validation_data, verbose=verbose)
            else:
                hist = self.model.fit_generator(x, y, batch_size=batch_size,
                                                epochs=epoch, verbose=verbose)

        self.train_loss = hist.history['loss']
        self.val_loss = hist.history['val_loss']
        self.train_acc = hist.history['acc']
        self.val_acc = hist.history['val_acc']

    def save_model(self, file):
        """Saves the model"""
        self.model.save(file)

    def save_hist(self, file):
        """Saves the training history (train_loss, val_loss, train_accuracy, val_accuracy) as txt file"""
        hist = (np.array(self.train_loss), np.array(self.val_loss), np.array(self.train_acc), np.array(self.val_acc))
        np.savetxt(file, hist, delimiter=',')

    def vis_acc_loss(self, file=None):
        """
        Plot test and validation loss and accuracy.

        Parameters:
            file: file name to create on disc, including full path
        """
        x_axis = np.arange(1, self.epoch+1, step=1, dtype=int)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        
        ax1.plot(x_axis, self.train_loss)
        ax1.plot(x_axis, self.val_loss)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.title.set_text('Loss')
        ax1.grid(True)
        ax1.legend(['Training', 'Validation'], fontsize=12)
        plt.tight_layout()
        
        np.set_printoptions(precision=1)
        ax2.plot(x_axis, self.train_acc)
        ax2.plot(x_axis, self.val_acc)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.title.set_text('Accuracy')
        plt.style.use(['classic'])
        ax2.grid(True)
        ax2.set_ylim([0.4, 1])
        ax2.set_yticks(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1]))
        ax2.set_yticklabels(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1]))
        ax2.set_xticks(x_axis)
        ax2.set_xticklabels(x_axis)
        plt.xlim([1, self.epoch])
        plt.tight_layout()
        if file:
            plt.savefig(file)
        else:
            plt.show()

    def vis_cm(self, x, y, classes, file_fig=None, file_mat=None):
        """
        Plot confusion matrix
        
        Parameters:
            x: test data
            y: corresponding labels
            classes: class labels
            file_fig: file name for confusion matrix figure, including full path
            file_mat: file name for confusion matrix values, including full path.
                Resulting file can be used to plot conf matrix with R code.
        """
        score = self.model.evaluate(x=x, y=y)
        # predictions
        y_pred = self.model.predict(x, batch_size=None, verbose=0)
        # confusion matrix
        y_pred_class = (y_pred > 0.5)
        cm = confusion_matrix(y, y_pred_class)

        if file_mat:
            np.savetxt(file_mat, (y_pred_class.astype(int)[:, 0], y), delimiter=',')

        np.set_printoptions(precision=2)
        plt.figure(facecolor='white')
        title = 'Normalized confusion matrix\nLoss=' + str(round(score[0], 2)) + ', Acc=' + str(round(score[1], 2))
        cmap = plt.cm.Blues
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        plt.title(title)
        plt.colorbar()
        plt.xticks([0, 1], classes, rotation=0, fontsize=12)
        plt.yticks([0, 1], classes, fontsize=12)
        plt.ylabel('True label', fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)
        plt.tight_layout()
        if file_fig:
            plt.savefig(file_fig)
        else:
            plt.show()


class MakeANN(MakeModel):
    """ Creates regular NeuralNet based classifier """
    
    def __init__(self, input_shape, lr, units, l2_regularizer, dropout, lr_dec, initializer='uniform'):
        """
        Creates a regular artificial neural network and compiles using Adam optimizer.
        
        Parameters:
            input_shape: input shape (number of summary statistics per simulation)
            units: a list of units of hidden layers
            l2_regularizer: l2 weight decay
            dropout: dropout rate (same for all fully connected layers)
            lr_dec: learning rate decay
            initializer: kernel initializer
        """
        # input layer
        inp_dim = (input_shape,)
        inp = Input(shape=inp_dim)
        x = Dense(units=units[0], kernel_initializer=initializer,
                  kernel_regularizer=regularizers.l2(l2_regularizer))(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
        # dense blocks
        for unit in units[1:]:
            x = super().make_dense_block(x, unit, l2_regularizer, dropout)
        # final classification layer
        x = Dense(1, activation='sigmoid')(x)
        # create model
        model = Model(inp, x, name='ann')

        model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                      optimizer=optimizers.Adam(lr=lr, decay=lr_dec))
        super().__init__(model)


class LoadANN(MakeModel):

    def __init__(self, test, input_shape=66):
        """
        Loads compiled ANN model with tuned hyperparameters
        
        Parameters:
            test: test number
            input_shape: input shape (number of summary statistics)
        """
        if test == 1:
            lr = 0.0005
            units = [20, 20, 10]
            l2_regularizer = 0.005
            dropout = 0.5
        elif test == 2:
            lr = 0.0005
            units = [20, 20, 10]
            l2_regularizer = 0.005
            dropout = 0.5
        elif test == 3:
            lr = 0.0005
            units = [20, 20, 10]
            l2_regularizer = 0.005
            dropout = 0.5
        else:
            raise ValueError("Test must be 1, 2, or 3")

        initializer = 'uniform'
        lr_dec = 0.00001

        # input layer
        inp_dim = (input_shape,)
        inp = Input(shape=inp_dim)
        x = Dense(units=units[0], kernel_initializer=initializer,
                  kernel_regularizer=regularizers.l2(l2_regularizer))(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
        # dense blocks
        for unit in units[1:]:
            x = super().make_dense_block(x, unit, l2_regularizer, initializer, dropout)
        # final classification layer
        x = Dense(1, activation='sigmoid')(x)
        # create model
        model = Model(inp, x, name='ann')

        model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                      optimizer=optimizers.Adam(lr=lr, decay=lr_dec))
        super().__init__(model)
          

class MakeCNN(MakeModel):
    """Creates ConvNet based classifier"""
    
    def __init__(self, input_shape, lr, lr_dec, filters, kernel_size, pooling_size, l2_regularizer,
                 units_fc=(128), batch_norm=True, dropout_conv=None, dropout_fc=0.5, initializer='uniform',
                 strides=(1, 1), padding='same'):
        """
        Initiates a convolutional neural network model.
        
        Parameters:
            input_shape: a tuple specifying input shape: (image_rows, image_cols)
            filters: number of filters in each conv layer
            kernel_size: kernel size (kernel_size, kernel_size)
            pooling_size: the size of maximum pooling
            initializer: kernel initializer to be used in the dense layer
            l2_regularizer: l2 weight decay
            batch_norm: if True (default) use batch normalization at each hidden layer
            units_fc: units at fully-connected layers
            dropout_conv: dropout rate at convolutional layers   
            dropout_fc: dropout rate at full-connected layers
        """
        # input layer
        inp = Input(shape=input_shape)
        x = Conv2D(filters[0], (kernel_size, kernel_size), strides=strides, padding=padding,
                   data_format='channels_last', kernel_regularizer=regularizers.l2(l2_regularizer),
                   kernel_initializer=initializer)(inp)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(pooling_size, pooling_size))(x)
        if dropout_conv:
            x = Dropout(dropout_conv)(x)

        for filtr in filters[1:]:
            x = super().make_conv_block(input_tensor=x, filters=filtr, kernel_size=kernel_size,
                                        pooling_size=pooling_size, l2_regularizer=l2_regularizer,
                                        dropout=dropout_conv, initializer=initializer, strides=strides,
                                        padding=padding, batch=batch_norm)
        x = Flatten()(x)
        # fully-connected layer
        for unit in units_fc:
            x = super().make_dense_block(x, unit, l2_regularizer, initializer=initializer,
                                         dropout=dropout_fc, batch=batch_norm)
        # classification layer
        x = Dense(1, activation='sigmoid')(x)
        
        model = Model(inp, x, name='cnn')

        model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                      optimizer=optimizers.Adam(lr=lr, decay=lr_dec))
        super().__init__(model)
        
        
class LoadCNN(MakeModel):

    def __init__(self, test, img_dim=(128, 128)):
        """
        Loads compiled CNN model with tuned hyperparameters
        Parameters:
            test: test number
            img_dim: image dimension (nrow, ncol)
        """
        if test == 1:
            lr = 0.00005
            dropout_fc = 0.0
            dropout_conv = 0.5
            l2_regularizer = 0.005
        elif test == 2:
            lr = 0.00005
            dropout_fc = 0.0
            dropout_conv = 0.5
            l2_regularizer = 0.005
        elif test == 3:
            lr = 0.0001
            dropout_fc = 0.5
            dropout_conv = 0.5
            l2_regularizer = 0.001
        else:
            raise ValueError("Test must be 1, 2, or 3")

        input_shape = (img_dim[0], img_dim[1], 1)
        lr_dec = 0.00001
        filters = [32, 32, 32]
        units_fc = [128]
        kernel_size = 3
        pooling_size = 2
        initializer = 'uniform'
        strides = (1, 1)
        padding = 'same'

        # input layer
        inp = Input(shape=input_shape)
        x = Conv2D(filters[0], (kernel_size, kernel_size), strides=strides, padding=padding,
                   data_format='channels_last', kernel_regularizer=regularizers.l2(l2_regularizer),
                   kernel_initializer=initializer)(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(pooling_size, pooling_size))(x)
        # x = Dropout(dropout_conv)(x)

        # conv layer #2
        x = super().make_conv_block(input_tensor=x, filters=filters[1], kernel_size=kernel_size,
                                    pooling_size=pooling_size, l2_regularizer=l2_regularizer,
                                    dropout=dropout_conv, initializer=initializer, strides=strides,
                                    padding=padding, batch=True, pooling=True)

        # conv layer #3
        x = super().make_conv_block(input_tensor=x, filters=filters[2], kernel_size=kernel_size,
                                    pooling_size=pooling_size, l2_regularizer=l2_regularizer,
                                    dropout=dropout_conv, initializer=initializer, strides=strides,
                                    padding=padding, batch=True, pooling=False)

        x = Flatten()(x)

        # fully-connected layers
        x = super().make_dense_block(x, units_fc[0], l2_regularizer, initializer=initializer,
                                     dropout=dropout_fc, batch=False)

        # x = super().make_dense_block(x, units_fc[1], l2_regularizer, initializer=initializer,
        #                 dropout = dropout_fc, batch = True)

        # classification layer
        x = Dense(1, activation='sigmoid')(x)
        # create model
        model = Model(inp, x, name='cnn')

        model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                      optimizer=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=lr_dec))
        super().__init__(model)


def predict(x, model, labels, test, file=None):
    """
    Performs prediction on real data

    Parameters:
        x: input data
        model: full path to the trained model
        labels: list of labels (SNP IDs)
        test: test number
        file: location where the output dataframe will be saved. If None, returns the data frame
    """
    if test == 1:
        mapings = {0: "Neutral", 1: "Selection"}
    elif test == 2:
        mapings = {0: "Incomplete Sweep", 1: "Balancing Selection"}
    elif test == 3:
        mapings = {0: "Overdominance", 1: "Negative Freq-Dep. Selection"}
    else:
        raise ValueError("Invalid test number {}".format(test))

    model = load_model(model)
    y_pred = model.predict(x, batch_size=None, verbose=0)
    y_pred_class = (y_pred > 0.5)
    y_pred_class = [mapings[i[0]] for i in np.array(y_pred_class, dtype="int32")]

    df = pd.DataFrame({"SNP": labels,
                       "Pred": list(chain(*y_pred)),
                       "PredClass": y_pred_class})
    if file:
        df.to_csv(file)
    else:
        return df