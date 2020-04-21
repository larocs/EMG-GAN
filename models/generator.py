#MIT License
#
#Copyright (c) 2017 Erik Linder-Norén
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
# Laboratory of Robotics and Cognitive Science
# Version by:  Rafael Anicet Zanini
# Github:      https://github.com/larocs/EMG-GAN

from keras.utils import plot_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Flatten, Activation, Dropout, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, GRU
from keras.layers import Add, Subtract, Multiply, ReLU, ThresholdedReLU, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalAvgPool1D
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import keras.backend as K
import warnings 
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

class Generator():

    def __init__(self, args, training = False):
        self.noise_dim = args['noise_dim']
        self.channels = args['channels']
        self.conv_activation = args['conv_activation']
        self.activation_function = args['activation_function']
        self.num_steps = args['num_steps']
        self.seq_shape = (self.num_steps, self.channels)
        self.sliding_window = args['sliding_window']
        self.training_mode = training
        self.model = self.build_generator()
    
    def moving_avg(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
    
        # Returns
            z (tensor): sampled latent vector
        """
        input_ = args
        sliding_window = tf.contrib.signal.frame(
            input_,
            self.sliding_window,
            1,#steps
            pad_end=True,
            pad_value=0,
            axis=1,
            name='envelope_moving_average'
        )
        sliding_reshaped = K.reshape(sliding_window,(-1,self.num_steps,self.sliding_window))
        mvg_avg = K.mean(sliding_reshaped, axis=2, keepdims=True)
        return mvg_avg
    
    def build_generator(self):
    
        model = Sequential()
        model.add(Reshape((self.noise_dim,self.channels), input_shape=(self.noise_dim,)))
        model.add(Conv1D(128, kernel_size=4, padding="same", data_format="channels_last"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation(self.conv_activation))
        model.add(UpSampling1D())
        model.add(Conv1D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation(self.conv_activation))
        model.add(UpSampling1D())
        model.add(Conv1D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation(self.conv_activation))
        model.add(UpSampling1D())
        model.add(Conv1D(32, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation(self.conv_activation))
        model.add(Conv1D(16, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation(self.conv_activation))
        model.add(Conv1D(self.channels, kernel_size=4, padding="same"))        
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation(self.conv_activation))
        model.add(Flatten())
        model.add(Dense(self.num_steps * self.channels))
        model.add(Activation(self.activation_function))
        model.add(Reshape((self.num_steps,self.channels)))
        
        if self.sliding_window > 0:
            model.add(Lambda(self.moving_avg, output_shape=self.seq_shape, name='mvg_avg'))
        
        if self.training_mode:
            print('Generator model:')
            model.summary()
            model_json = model.to_json()
            
            with open('./output/generator.json', "w") as json_file:
                json_file.write(model_json)
                
            file_name = './output/generator.png'
            plot_model(model, to_file=file_name, show_shapes = True)        
    
        return model
    
    def save(self, index=-1):
        if index == -1:
            file_path = './saved_models/generator.h5'
        else:
            file_path = './saved_models/generator_' + str(index) + '.h5'
        self.model.save_weights(file_path)
    
    def load(self, index=-1):
        if index == -1:
            file_path = './saved_models/generator.h5'
        else:
            file_path = './saved_models/generator_' + str(index) + '.h5'
        self.model = self.build_generator()
        self.model.load_weights(file_path)
    
    def predict(self, args):
        return self.model.predict(args)
    