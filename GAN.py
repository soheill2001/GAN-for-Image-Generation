import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import argparse
from Generator import Generator
from Discriminator import Discriminator

@tf.function
def train_step(images, generator, discriminator, generator_optimizer, discriminator_optimizer, batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator.model(noise, training=True)
        real_output = discriminator.model(images, training=True)
        fake_output = discriminator.model(generated_images, training=True)
        gen_loss = generator.loss(fake_output)
        disc_loss = discriminator.loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.model.trainable_variables))
    return gen_loss, disc_loss

def preprocess_image(image, label):
    image = tf.image.resize(image, [64, 64])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def show_image(generator, epoch, train=False):
    noise = tf.random.normal([16, 100])
    generated_images = generator.model(noise, training=False)
    fig = plt.figure(figsize=(5, 5))
    for i in range(generated_images.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(generated_images[i, :, :] * 0.5 + 0.5)
      plt.axis('off')
    if train:
        plt.savefig(f'images/epoch_{epoch}.jpg')
    else:
      if not os.path.exists('test'):
        os.makedirs('test')
      plt.savefig(f'test/output.jpg')

def save_samples(generator, epoch, noise_dim):
    if not os.path.exists('images'):
        os.makedirs('images')
    show_image(generator, epoch, train=True)

def show_losses(G_loss, D_loss):
  plt.figure(figsize=(10,5))
  plt.title("Generator and Discriminator Loss")
  plt.plot(G_loss, label="G")
  plt.plot(D_loss, label="D")
  plt.xlabel("iterations")
  plt.ylabel("Loss")
  plt.legend()
  if not os.path.exists('loss'):
    os.makedirs('loss')
  plt.savefig(f'loss/loss.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--generator_path', type=str, default="")
    parser.add_argument('--discriminator_path', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    if args.generator_path != "":
        generator = Generator(model_path=args.generator_path)
        discriminator = Discriminator(model_path=args.discriminator_path)
        show_image(generator, args.epochs)
    else:
        generator = Generator(model_path=args.generator_path)
        discriminator = Discriminator(model_path=args.discriminator_path)
        generator_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5, 0.999)
        discriminator_optimizer = tf.keras.optimizers.Adam(0.0002, 0.5, 0.999)
        noise_dim = 100
        dataset, info = tfds.load('tf_flowers', with_info=True, as_supervised=True)
        dataset = dataset['train'].map(preprocess_image).shuffle(10000).batch(args.batch_size)
        G_loss = []
        D_loss = []
        for epoch in range(args.epochs):
          for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, generator, discriminator, generator_optimizer, discriminator_optimizer, args.batch_size, noise_dim)
          G_loss.append(gen_loss)
          D_loss.append(disc_loss)
          print(f'Epoch {epoch+1}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}')
          if (epoch+1) % 10 == 0:
            save_samples(generator, epoch+1, noise_dim)
        show_losses(G_loss, D_loss)
        generator.model.save("Gen.h5")
        discriminator.model.save("Dis.h5")