import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = keras.layers.Dense(8*8*256, use_bias=False, input_shape=(100,), name='Dense')
        self.rshpe_lyr = keras.layers.Reshape((8, 8, 256), name='Reshape1')

        # (None,8,8,128)
        self.convt1 = keras.layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False, name='ConvT1')
        self.bn1 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm1')
        self.actv1 = keras.layers.LeakyReLU(name='ReLU1')

        # (None, 32, 32, 64)
        self.convt2 = keras.layers.Conv2DTranspose(64, (5,5), strides=(4,4), padding='same', use_bias=False, name='ConvT2')
        self.bn2 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm2')
        self.actv2 = keras.layers.LeakyReLU(name='ReLU2')

        # (None, 64, 64, 64)
        self.convt3 = keras.layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False, name='ConvT3')
        self.bn3 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm3')
        self.actv3 = keras.layers.LeakyReLU(name='ReLU3')
        
        # (None, 128, 128, 1)
        self.convt4 = keras.layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, name='ConvT4', activation='tanh')


    def call(self, input):
        x = self.dense(input)
        x = self.rshpe_lyr(x)
        # assert x.shape == (None, 8, 8, 256)

        x = self.convt1(x)
        x = self.bn1(x)
        x = self.actv1(x)

        x = self.convt2(x)
        x = self.bn2(x)
        x = self.actv2(x)

        x = self.convt3(x)
        x = self.bn3(x)
        x = self.actv3(x)

        x = self.convt4(x)
        
        return x