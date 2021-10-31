import tensorflow as tf
from tensorflow import keras

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = keras.layers.Dense(8*8*128, use_bias=False, input_shape=(100,), name='Dense')
        self.rshpe_lyr = keras.layers.Reshape((8, 8, 128), name='Reshape1')

        self.upsample1 = keras.layers.UpSampling2D(name='UpSample2')
        self.conv1 = keras.layers.Conv2D(64, kernel_size=3, padding='same', name='Conv2')
        self.bn1 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm1')
        self.actv1 = keras.layers.LeakyReLU(name='ReLU1')

        self.upsample2 = keras.layers.UpSampling2D(name='UpSample2')
        self.conv2 = keras.layers.Conv2D(32, kernel_size=3, padding='same', name='Conv2')
        self.bn2 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm2')
        self.actv2 = keras.layers.LeakyReLU(name='ReLU2')

        self.upsample3 = keras.layers.UpSampling2D(name='UpSample3')
        self.conv3 = keras.layers.Conv2D(32, kernel_size=3, padding='same', name='Conv3')
        self.bn3 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm3')
        self.actv3 = keras.layers.LeakyReLU(name='ReLU3')
        
        self.upsample4 = keras.layers.UpSampling2D(name='UpSample4')
        self.conv4 = keras.layers.Conv2D(1, kernel_size=3, padding='same', name='Conv4')
        self.bn4 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm4')
        self.actv4 = keras.layers.Activation('tanh')


    def call(self, input):
        x = self.dense(input)
        x = self.rshpe_lyr(x)

        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.actv1(x)

        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.actv2(x)

        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.actv3(x)

        x = self.upsample4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.actv4(x)
        
        return x