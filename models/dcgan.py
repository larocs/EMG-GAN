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


from __future__ import print_function, division

from keras.optimizers import RMSprop, Adam
from keras import Model, Sequential
from keras.layers import Input
import keras.backend as K
import matplotlib.pyplot as plt
import sys
import datetime
import numpy as np
import pandas as pd
import pathlib

from utils.data_utils import DataLoader
from utils.plot_utils import *
from models.generator import Generator
from models.discriminator import Discriminator

import warnings 
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf


class DCGAN():
    def __init__(self, args, training = False):
        
        # Define parameters for training
        self.learning_rate = args["learning_rate"]  # Learning rate
        self.loss_function = args["loss_function"]  # loss_function
        self.metrics = args["metrics"] # metrics
        self.channels = args['channels']
        self.batch_size = args['batch_size']
        self.num_steps = args['num_steps']
        self.seq_shape = (self.num_steps, self.channels)
        self.noise_dim = args["noise_dim"] # metrics
        self.use_random_noise = args["use_random_noise"]
        self.training_mode = training

        # Following parameter and optimizer set as recommended in paper
        self.optimizer = Adam(lr=self.learning_rate)

        # Build and compile the critic
        self.critic = Discriminator(args, training)
        self.critic.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)

        # Build the generator
        self.generator = Generator(args, training)
        
        # Build de combined model (generator => critic)

        # The generator takes latent space as input and generates fake signal as output
        z = Input(shape=(self.noise_dim,))
        signal = self.generator.model(z)

        # For the combined model we will only train the generator
        self.critic.model.trainable = False
        
        # The critic takes generated signal as input and determines validity
        valid = self.critic.model(signal)

        # The combined model  (stacked generator and critic) - try to minimize the loss
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=self.optimizer)
    
    def save_critic(self, index = -1):
        self.critic.save(index)
    
    def save_generator(self, index = -1):
        self.generator.save(index)
    
    def load(self, index = -1):
        self.generator.load(index)
        self.critic.load(index)

    
    def save_sample(self, epoch, signals):
        
        # Generate latent noise
        noise = self.generate_noise(signals)
        
        # Generate a batch of new EMG signals
        gen_signal = self.generator.predict(noise)
        
        #Evaluate with critic
        critic_signal = self.critic.predict(gen_signal)
        
        #Reshape for saving on csv
        gen_signal = np.reshape(gen_signal, (gen_signal.shape[0],gen_signal.shape[1]))
        critic_signal = np.reshape(critic_signal, (critic_signal.shape[0],critic_signal.shape[1]))
        np.savetxt('./output/Noise_' + str(epoch) + '.csv', noise, delimiter=",")
        np.savetxt('./output/Generated_' + str(epoch) + '.csv', gen_signal, delimiter=",")
        np.savetxt('./output/Validated_' + str(epoch) + '.csv', critic_signal, delimiter=",")
        
        #Plot the reference signal
        ref_signal = signals[:,:,:]
        ref_signal = np.reshape(ref_signal, (ref_signal.shape[0],ref_signal.shape[1]))
        plot_reference(ref_signal, epoch)
        np.savetxt('./output/Reference_' + str(epoch) + '.csv', ref_signal, delimiter=",")
        
        #Plot the generated signals with epoch
        plot_prediction(gen_signal, epoch)
        return gen_signal

    def generate_noise(self, signals):
        # Sample noise as generator input
        if self.use_random_noise:
            #Generate random distribution between interval [-1.0,1.0]
            #noise = 2.0 * np.random.random_sample((self.batch_size,self.noise_dim)) -1.0
        
            #Alternative - use senoid as input also is possible, and give good results
            x = np.linspace(-np.pi, np.pi, self.noise_dim)
            noise = 0.1 * np.random.random_sample((self.batch_size,self.noise_dim)) + 0.9 * np.sin(x)
        # Sample input data as generator input
        else:
            noise = signals[:,0:self.noise_dim,:]
            noise = np.reshape(noise,(noise.shape[0],noise.shape[1]))
        return noise