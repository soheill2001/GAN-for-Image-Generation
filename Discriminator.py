from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, BatchNormalization
from tensorflow.keras.models import load_model

class Discriminator():
    def __init__(self, model_path=""):
        if model_path != "":
            self.Load_Model(model_path)
        else:
            self.model = Sequential()
            self.build_model()
            
    def build_model(self):
        self.model.add(Conv2D(64, kernel_size=4, strides=2, padding='same', input_shape=(64, 64, 3)))
        self.model.add(LeakyReLU(0.2))
        self.model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU(0.2))
        self.model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU(0.2))
        self.model.add(Conv2D(512, kernel_size=4, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))

    def loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss
    
    def Load_Model(self, model_path):
        self.model = load_model(model_path)