from tensorflow import keras

class Generator(keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.latent_dim = latent_dim

        self.dense = keras.layers.Dense(8*16*256, use_bias=False, input_shape=(self.latent_dim,), name='Dense')
        self.rshpe_lyr = keras.layers.Reshape((8, 16, 256), name='Reshape1')

        # (None,8,16,128)
        self.convt1 = keras.layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False, name='ConvT1', kernel_initializer='he_normal')
        self.bn1 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm1')
        self.actv1 = keras.layers.LeakyReLU(name='ReLU1')

        # (None, 16, 32, 64)
        self.convt2 = keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False, name='ConvT2', kernel_initializer='he_normal')
        self.bn2 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm2')
        self.actv2 = keras.layers.LeakyReLU(name='ReLU2')

        # (None, 32, 64, 64)
        self.convt3 = keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False, name='ConvT3', kernel_initializer='he_normal')
        self.bn3 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm3')
        self.actv3 = keras.layers.LeakyReLU(name='ReLU3')

        # (None, 64, 128, 64)
        self.convt4 = keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False, name='ConvT4', kernel_initializer='he_normal')
        self.bn4 = keras.layers.BatchNormalization(momentum=0.8, name='BatchNorm4')
        self.actv4 = keras.layers.LeakyReLU(name='ReLU4')
        
        # (None, 128, 256, 1)
        self.convt5 = keras.layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, name='ConvT5', activation='tanh', kernel_initializer='he_normal')


    def call(self, input):
        x = self.dense(input)
        x = self.rshpe_lyr(x)

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
        x = self.bn4(x)
        x = self.actv4(x)

        x = self.convt5(x)
        
        return x