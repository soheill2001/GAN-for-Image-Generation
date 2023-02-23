from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape,Conv2DTranspose, BatchNormalization, ReLU
from tensorflow.keras.models import load_model

class Generator():

    def __init__(self, model_path=""):
      if model_path != "":
          self.Load_Model(model_path)
      else:
        self.model = Sequential()
        self.build_model()

    def build_model(self):
        self.model.add(Dense(4*4*512, input_shape=(100,)))
        self.model.add(Reshape((4, 4, 512)))
        self.model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(ReLU())
        self.model.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))

    def loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def Load_Model(self, model_path):
        self.model = load_model(model_path)