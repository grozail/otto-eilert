import json
import numpy as np
from timeit import default_timer as timer
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import os.path

model_file = 'mnist_model.h5'

if __name__ == '__main__':
    img_rows, img_cols = 28, 28
    
    n_filters = 32
    
    pool_size = (2, 2)
    
    kernel_size = (3, 3)
    
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    
    input_shape = (img_rows, img_cols, 1)
    
    X_train.astype('float32')
    X_test.astype('float32')
    
    X_train = np.divide(X_train, 255.0)
    X_test = np.divide(X_test, 255.0)
    
    n_classes = 10
    
    Y_train = np_utils.to_categorical(y_train, n_classes)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    
    if os.path.isfile(model_file):
        model = load_model(model_file)
    else:
        model = Sequential()
        model.add(Convolution2D(n_filters, kernel_size, padding='valid', input_shape=input_shape))
        model.add(Activation('relu'))
        
        model.add(Convolution2D(n_filters, kernel_size))
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))
        model.add(Flatten())
        
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(n_classes))
        model.add(Activation('softmax'))
        
        model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        
        batch_size = 128
        n_epochs = 1
        
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=n_epochs, verbose=1, validation_data=(X_test, Y_test))
        
        score = model.evaluate(X_test, Y_test, verbose=2)
        
        model.save(model_file)
        
        plot_model(model, to_file='mnist_model.png', show_shapes=True)