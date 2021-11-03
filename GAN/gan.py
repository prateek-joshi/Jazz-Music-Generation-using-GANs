import tensorflow as tf 
from tensorflow import keras
import numpy as np
tf.config.run_functions_eagerly(True)


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator 
        self.latent_dim = latent_dim
        
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = keras.losses.get(loss_fn)
    
    def train_step(self, data):
        real_images = data
        tf.cast(real_images, dtype=tf.float32)
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float32)   # Gaussian Noise
        
        generated_images = self.generator(random_latent_vectors)
        combined_images = tf.concat([generated_images.numpy(), real_images.numpy()], axis=0)
        
        labels = tf.concat([tf.ones((batch_size, 1)),
                            tf.zeros((batch_size, 1))], axis=0)
        
        labels += 0.05 * tf.random.uniform(tf.shape(labels))
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
            
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
        
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.zeros((batch_size, 1))
        
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
            
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        return {"discriminator_loss": d_loss, "generator_loss": g_loss}