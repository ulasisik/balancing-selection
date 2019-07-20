#!/usr/bin/env python3.6
"""
Contains required modules to create, train and evaluate deep neural network based classifier

@author: ulas isildak
"""
import sys 
sys.path.insert(0, '/mnt/NAS/projects/2018_deepLearn_selection/50kb/balancing-selection/scripts/BaSe') #path to BaSe package

import numpy as np

import matplotlib
matplotlib.use('Agg')   #or use ssh -X to connect to sever
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
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


class MakeModel(object):
    '''
    Abstract class for creating models and visualizing result
    '''
    
    
    @staticmethod
    def make_dense_block(input_tensor, unit, l2_regularizer, initializer, 
                     dropout, batch=True):
        '''
        Creates a hidden layer
        
        Args:
            input_tensor: input tensor
            initilizer: kernel initializer to be used in the dense layer
            l2_regularizer: l2 weight decay
            unit: number of neurons(units) in the dense layer
            dropout: dropout rate
        '''
        
        x = Dense(unit, kernel_initializer = initializer, kernel_regularizer = regularizers.l2(l2_regularizer))(input_tensor)
        if batch:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout:
            x = Dropout(dropout)(x)
        
        return x


    @staticmethod
    def make_conv_block(input_tensor, filters, kernel_size, pooling_size, 
                        l2_regularizer, initializer, strides,
                        padding, batch=True, dropout=None, pooling=True):
        '''
        Creates a convolutional block
        
        Args:
            inpu_tensor: input tensor
            filters: number of filter
            kernel_size: kernel size (kernel_size, kernel_size)
            pooling_size: the size of maximum pooling
            initilizer: kernel initializer 
            l2_regularizer: l2 weight decay
            dropout: dropout rate             
        '''
        
        x = Conv2D(filters,(kernel_size,kernel_size), strides=strides, padding=padding, 
                          data_format='channels_last', kernel_regularizer=regularizers.l2(l2_regularizer),
                          kernel_initializer=initializer)(input_tensor)
        if batch:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if pooling:
            x = MaxPooling2D(pool_size=(pooling_size,pooling_size))(x)
        if dropout:
            x = Dropout(dropout)(x)
        
        return x
        
        
    def compile_model(self, lr, lr_dec):
        '''
        Compiles cnn model using Adam optimizer.
        
        Args:
            lr: learning rate
            lr_dec: learning rate decay
        '''
        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=lr_dec))
    
        
    def fit_model(self, X, y, batch_size, epoch, aug=False, validation_data=None, file_model = None, file_hist = None):
        '''
        Fits model to the data
        
        Args:
            X: training set
            y: corresponding labels
            batch_size: batch size
            epoch: epoch
            aug: if True, performs image agumentation. Default is False.
            validation_data: validation data (X_val, y_val)
            file_model: file name to save full model, including full path (.h5)
            file_hist: file name to save training history including train_loss, 
                val_loss, train_acc, train_loss
        '''
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

            datagen_train.fit(X)

            if validation_data:
                X_val, y_val = validation_data

                datagen_val = ImageDataGenerator(
                    samplewise_center=False,  # set each sample mean to 0
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images

                datagen_val.fit(X_val)

                hist = self.model.fit_generator(datagen_train.flow(X, y, batch_size=batch_size),
                        validation_data=datagen_val.flow(X_val, y_val, batch_size=batch_size),
                        epochs = epoch, verbose=2)

            else:
                hist= self.model.fit_generator(datagen.flow(X, y, batch_size=batch_size), 
                                      epochs=epoch, validation_data=(X_val, y_val), verbose=2)

        else:
            hist = self.model.fit(X, y, batch_size = batch_size, epochs = epoch, 
                    validation_data = validation_data, verbose = 2)
        
        if file_model:
            self.model.save(file_model)
        if file_hist:
            train_loss=hist.history['loss']
            val_loss=hist.history['val_loss']
            train_acc=hist.history['acc']
            val_acc=hist.history['val_acc']
            
            np.savetxt(file_hist, (np.array(train_loss), np.array(val_loss), np.array(train_acc), np.array(val_acc)), delimiter=',')

        self.train_loss=hist.history['loss']
        self.val_loss=hist.history['val_loss']
        self.train_acc=hist.history['acc']
        self.val_acc=hist.history['val_acc']


    def vis_acc_loss(self, file_fig):
        '''
        Plot test and validation loss and accuracy.

        Args:
            file_fig: file name to create on disc, including full path
        '''
        
        x_axis = np.arange(1, self.epoch+1, step = 1, dtype=int)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
        
        ax1.plot(x_axis,self.train_loss)
        ax1.plot(x_axis,self.val_loss)
        ax1.set_ylabel('Loss',fontsize=12)
        ax1.title.set_text('Loss')
        ax1.grid(True)
        ax1.legend(['Training','Validation'],fontsize=12)
        #plt.style.use(['classic'])
        plt.tight_layout()
        
        np.set_printoptions(precision=1)
        ax2.plot(x_axis,self.train_acc)
        ax2.plot(x_axis,self.val_acc)
        ax2.set_ylabel('Accuracy',fontsize=12)
        ax2.set_xlabel('Epoch',fontsize=12)
        ax2.title.set_text('Accuracy')
        plt.style.use(['classic'])
        ax2.grid(True)
        ax2.set_ylim([0.5, 1])
        ax2.set_yticks(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1]))
        ax2.set_yticklabels(np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1]))
        ax2.set_xticks(x_axis)
        ax2.set_xticklabels(x_axis)
        plt.xlim([1,self.epoch])
        plt.tight_layout()
        plt.savefig(file_fig)
        
    
    def vis_cm(self, X, y, file_fig, file_mat=None):
        '''
        Plot confusion matrix
        
        Args:
            X: test data
            y: corresponding labels
            file_fig: file name for confusion matrix figure, including full path
            file_mat: file name for confusion matrix values, including full path.
                Resulting file can be used to plot conf matrix with R code.
        '''
        score = self.model.evaluate(x=X, y=y)
        #predictions
        y_pred = self.model.predict(X, batch_size=None, verbose=1)
        
        #confusion matrix
        y_pred_class=(y_pred>0.5)
        if file_mat:
            np.savetxt(file_mat, (y_pred_class.astype(int)[:,0], y), delimiter=',')
        cm = confusion_matrix(y, y_pred_class)
        
        np.set_printoptions(precision=2)
        plt.figure(facecolor='white')
        title='Normalized confusion matrix\nLoss=' +  str(round(score[0],2)) + ', Acc=' + str(round(score[1],2))
        cmap=plt.cm.Blues
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        plt.title(title)
        plt.colorbar()
        plt.xticks([0,1], rotation=0, fontsize=12)
        plt.yticks([0,1], fontsize=12)
        plt.ylabel('True label', fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)
        plt.tight_layout()
        plt.savefig(file_fig)


class MakeANN(MakeModel):
    '''
    Creates regular NeuralNet based classifier
    '''
    
    def __init__(self, input_shape, units, l2_regularizer, dropout = 0.5, initializer = 'uniform'):
        '''
        Initiates a regular neural network model.
        
        Args:
            units: a list of units of hidden layers
            initilizer: kernel initializer 
            l2_regularizer: l2 weight decay
            dropout: dropout rate     
            path_to_save: path to a directory where the model and data will be saved
        '''
        #input layer
        inp_dim = (input_shape,)
        inp = Input(shape=inp_dim)
        x = Dense(units = units[0], kernel_initializer = initializer, 
                  kernel_regularizer = regularizers.l2(l2_regularizer))(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
        #dense blocks
        for unit in units[1:]:
            x = super().make_dense_block(x, unit, l2_regularizer, dropout)
        #final classification layer
        x = Dense(1, activation = 'sigmoid')(x)
        #create model
        model = Model(inp, x, name='ann')
        
        self.model = model


class LoadANN(MakeModel):
    
    def __init__(self, test):
        '''
        Loads compiled ANN model with tuned hyperparameters
        
        Args:
            input_shape: input shape
        '''
        initializer = 'uniform'
        if test == 1:
            units = [20, 15, 10]
            l2_regularizer = 0.0005
            lr = 0.005
            lr_dec = 0.001
            dropout = 0.2
        elif test == 2:
            units = [20, 20, 10]
            l2_regularizer = 0.0001
            lr = 0.0005
            lr_dec = 0.00001
            dropout = 0.2
        elif test == 3:
            units = [20, 20, 10]
            l2_regularizer = 0.001
            lr = 0.0005
            lr_dec = 0.0001
            dropout = 0.5
        
        #input layer
        inp_dim = (66,)
        inp = Input(shape=inp_dim)
        x = Dense(units = units[0], kernel_initializer = initializer, 
                  kernel_regularizer = regularizers.l2(l2_regularizer))(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
        #dense blocks
        for unit in units[1:]:
            x = super().make_dense_block(x, unit, l2_regularizer, initializer, dropout)
        #final classification layer
        x = Dense(1, activation = 'sigmoid')(x)
        #create model
        model = Model(inp, x, name='ann')
       
        model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=optimizers.Adam(lr=lr, decay=lr_dec))
 
        self.model = model
          

class MakeCNN(MakeModel):
    'Creates ConvNet based classifier'
    
    def __init__(self, input_shape, filters, kernel_size, pooling_size, l2_regularizer, units_fc = [128], batch_norm = True,
                 dropout_conv = None, dropout_fc = 0.5, initializer = 'uniform', strides = (1,1), padding = 'same'):
        '''
        Initiates a convolutional neural network model.
        
        Args:
            input_shape: a tuple specifying input shape: (image_rows, image_cols)
            filters: number of filters in each conv layer
            kernel_size: kernel size (kernel_size, kernel_size)
                        pooling_size: the size of maximum pooling
            initilizer: kernel initializer to be used in the dense layer
            l2_regularizer: l2 weight decay
            batch_norm: if True (default) use batch normalization at each hidden layer
            units_fc: units at fully-connected layers
            dropout_conv: dropout rate at convolutional layers   
            dropout_fc: dropout rate at full-connected layers
        '''
        #input layer
        inp = Input(shape = input_shape)
        x = Conv2D(filters[0],(kernel_size,kernel_size), strides=strides, padding=padding, 
                          data_format='channels_last', kernel_regularizer=regularizers.l2(l2_regularizer),
                          kernel_initializer=initializer)(inp)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(pooling_size,pooling_size))(x)
        if dropout_conv:
            x = Dropout(dropout_conv)(x)

        for filtr in filters[1:]:
            x = super().make_conv_block(input_tensor=x, filters=filtr, kernel_size=kernel_size, 
                                pooling_size=pooling_size, l2_regularizer=l2_regularizer, 
                                dropout=dropout_conv, initializer=initializer, strides=strides,
                                padding=padding, batch=batch_norm)
        x = Flatten()(x)
        #fully-connected layer
        for unit in units_fc:
            x = super().make_dense_block(x, unit, l2_regularizer, initializer=initializer,
                                dropout = dropout_fc, batch = batch_norm)
        #classification layer
        x = Dense(1, activation='sigmoid')(x)
        
        model = Model(inp, x, name='cnn')
        
        self.model = model
        
        
class LoadCNN(MakeModel):
  
    def __init__(self, test, selection_start):
        '''
        Loads compiled CNN model with tuned hyperparameters
        
        '''
        #params
        if test == 1:
            lr_dec = 0.00001
            if selection_start == 'all':
                lr = 0.0005
                dropout_fc = 0.8
                dropout_conv = 0.5
            else:
                lr = 0.0001
                dropout_fc = 0.5
                dropout_conv = 0.5
        elif test == 2:
            lr_dec = 0.00001
            if selection_start == 'all':
                lr = 0.0005
                dropout_fc = 0.8
                dropout_conv = 0.5
            else:
                lr = 0.0001
                dropout_fc = 0.5
                dropout_conv = 0.5
        elif test == 3:
            lr_dec = 0.00001
            if selection_start == 'all':
                lr = 0.0005
                dropout_fc = 0.8
                dropout_conv = 0.5
            else:
                lr = 0.0001
                dropout_fc = 0.8
                dropout_conv = 0.5

        filters = [32, 32, 32]
        input_shape = (128, 128, 1)
        units_fc = [128]
        l2_regularizer = 0.01
        kernel_size = 3
        pooling_size = 2
        initializer = 'uniform' 
        strides = (1,1)
        padding = 'same'
        
        #input layer
        inp = Input(shape = input_shape)
        x = Conv2D(filters[0],(kernel_size,kernel_size), strides=strides, padding=padding, 
                          data_format='channels_last', kernel_regularizer=regularizers.l2(l2_regularizer),
                          kernel_initializer=initializer)(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(pooling_size,pooling_size))(x)
      #  x = Dropout(dropout_conv)(x)
        
        #conv layer #2
        x = super().make_conv_block(input_tensor=x, filters=filters[1], kernel_size=kernel_size, 
                          pooling_size=pooling_size, l2_regularizer=l2_regularizer, 
                          dropout=dropout_conv, initializer=initializer, strides=strides,
                          padding=padding, batch = True, pooling = True)
                            
        #conv layer #3
        x = super().make_conv_block(input_tensor=x, filters=filters[2], kernel_size=kernel_size, 
                          pooling_size=pooling_size, l2_regularizer=l2_regularizer, 
                          dropout=dropout_conv, initializer=initializer, strides=strides,
                          padding=padding, batch = True, pooling = False)
                  
        x = Flatten()(x)
        
        #fully-connected layers
        x = super().make_dense_block(x, units_fc[0], l2_regularizer, initializer=initializer,
                        dropout = dropout_fc, batch = True)
                        
        # x = super().make_dense_block(x, units_fc[1], l2_regularizer, initializer=initializer,
        #                 dropout = dropout_fc, batch = True)
                      

        #classification layer
        x = Dense(1, activation='sigmoid')(x)
        #create model
        model = Model(inp, x, name='cnn_test2')
        
        model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=lr_dec))
        
        self.model = model
        

