from tensorflow import keras
import tensorflow as tf

class Discriminator(keras.models.Model):
    def __init__(self, **kwargs) -> None:
        super(Discriminator, self).__init__(**kwargs)
        # coerce inputs into [0,1] space
        # self.norm = keras.layers.Normalization(axis=None, name='Normalize')

        self.conv1 = keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=(128, 256, 1), kernel_initializer='he_normal',name='Conv1')
        # self.pool1 = keras.layers.MaxPool2D(name='MaxPool1')
        self.actv1 = keras.layers.LeakyReLU(name='ReLU1')

        self.conv2 = keras.layers.Conv2D(128, (5,5), strides=(2,2), padding='same', kernel_initializer='he_normal', name='Conv2')
        # self.pool2 = keras.layers.MaxPool2D(name='MaxPool2')
        self.actv2 = keras.layers.LeakyReLU(name='ReLU2')

        self.conv3 = keras.layers.Conv2D(128, (5,5), strides=(2,2), padding='same', kernel_initializer='he_normal', name='Conv3')
        # self.pool2 = keras.layers.MaxPool2D(name='MaxPool2')
        self.actv3 = keras.layers.LeakyReLU(name='ReLU3')

        self.conv4 = keras.layers.Conv2D(128, (5,5), strides=(2,2), padding='same', kernel_initializer='he_normal', name='Conv4')
        # self.pool2 = keras.layers.MaxPool2D(name='MaxPool2')
        self.actv4 = keras.layers.LeakyReLU(name='ReLU4')

        self.flatten = keras.layers.Flatten(name='Flatten')
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.op = keras.layers.Dense(1, activation='softmax')

    def call(self, input, training=False):
        # x = self.norm(input)
        x = self.conv1(input)
        # x = self.pool1(x)
        x = self.actv1(x)
        if training:
            x = tf.nn.dropout(x, 0.3)
        
        x = self.conv2(x)
        # x = self.pool2(x)
        x = self.actv2(x)
        if training:
            x = tf.nn.dropout(x, 0.3)

        x = self.conv3(x)
        # x = self.pool2(x)
        x = self.actv3(x)
        if training:
            x = tf.nn.dropout(x, 0.3)

        x = self.conv4(x)
        # x = self.pool2(x)
        x = self.actv4(x)
        if training:
            x = tf.nn.dropout(x, 0.3)

        x = self.flatten(x)
        x = self.dense1(x)
        # x = self.dense2(x)

        return self.op(x)
