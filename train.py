import json, os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow import keras
import gan
import datetime
import cv2

parser = argparse.ArgumentParser(description='Sript to train the GAN model.')
parser.add_argument('--json_path', required=True, help='Path to json file.')
parser.add_argument('--epochs', required=False, default=100, help='Number of epochs to train the model.')
parser.add_argument('--save_weights', required=False, default=False, help='If set to true, saves model weights as checkpoints.')

args = parser.parse_args()

JSON_PATH = args.json_path
EPOCHS = int(args.epochs)
SAVE_WEIGHTS = bool(args.save_weights)
LATENT_DIM = 100
BATCH_SIZE = 32
SAVE_EPOCHS = 50    # Checkpoint after every 50 epochs

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    data = np.array(data['sgram'])
    return data

def preprocess(data_path, batch_size):
    dataset = load_data(data_path)
    data = tf.map_fn(lambda x: x/np.max(x), dataset, dtype=tf.double)
    data = data[...,np.newaxis]

    return tf.data.Dataset.from_tensor_slices(data).shuffle(data.shape[0]).batch(batch_size)

def train_model(data_path,  latent_dim, epochs, batch_size=32, save_weights=False, save_model=False):
    dataset = preprocess(data_path=data_path, batch_size=batch_size)

    generator_optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0005)

    discriminator = gan.discriminator.Discriminator()
    generator = gan.generator.Generator()
    ganModel = gan.gan.GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    ganModel.compile(discriminator_optimizer, generator_optimizer, keras.losses.BinaryCrossentropy(from_logits=False))

    callbacks = []
    if save_weights:
        checkpoint_savepath = os.path.join('data','saved_models','ckpt-{epoch:02d}')
        checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_savepath, save_weights_only=True, monitor=['generator_loss'], mode='min', save_freq=10*SAVE_EPOCHS)   # 10 steps per epoch
        callbacks.append(checkpoint_callback)

    _history = ganModel.fit(dataset, epochs=epochs, callbacks=callbacks)
    # _history = ganModel.fit(dataset, epochs=epochs)

    if save_model:
        generator.save('data','saved_models',f'generator-{datetime.now().strftime("%Y%m%d%H%M%S")}', save_format='tf')
        discriminator.save('data','saved_models',f'discriminator-{datetime.now().strftime("%Y%m%d%H%M%S")}', save_format='tf')

    return generator, discriminator


if __name__=='__main__':
    generator, discriminator = train_model(JSON_PATH, LATENT_DIM, EPOCHS, batch_size=BATCH_SIZE, save_weights=True)
    random_latent_vectors = tf.random.normal(shape=(1, LATENT_DIM), dtype=tf.float32)
    
    image = generator(random_latent_vectors)
    cv2.imwrite('.\\data\\images\\image.png', image)