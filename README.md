# Generative Adversarial Networks (GANs) for Image Generation
This code implements a Generative Adversarial Network (GAN) using TensorFlow, a popular deep learning framework. A GAN is a type of neural network that learns to generate new data that is similar to the training data. The model consists of two parts: a generator and a discriminator.

The generator creates new samples of data from random noise vectors, while the discriminator tries to distinguish between the generated samples and real data. These two networks are trained together in a game-theoretic manner, with the goal of improving the generator's ability to create realistic data and the discriminator's ability to distinguish between real and fake data.

## Prerequisites
+ Python
+ TensorFlow
+ Matplotlib
+ NumPy
+ argparse

you can install these libraries using the foloowing code
```
pip iinstall -r requirments.txt
```

## Usage
###Training

To train the GAN, run the GAN.py script. The script takes the following arguments:

+ --epochs: The number of epochs to train the GAN (default is 200).
+ --batch_size: The batch size to use for training (default is 128).

For example, to train the GAN for 500 epochs with a batch size of 256, run:
```
python GAN.py --epochs 500 --batch_size 256

```
This will train Generator and Discriminator and save images for each 10 epoch at `images` folder.
Also it will save loss of Generator and Discriminator at `loss` folder.
### Testing
To generate images using a pre-trained GAN, run the GAN.py script. The script takes the following arguments:

--num_images: The number of images to generate (default is 16).
--generator_path: The path to load the pre-trained generator model (default is "").
--discriminator_path: The path to load the pre-trained discriminator model (default is "").
For example, to generate a image using a pre-trained generator, run:
```
python GAN.py --generator_path Gen.h5 --discriminator_path Dis.h5
```
This will produce an image and save it at `test` folder.

## Results
After training for 500 epochs with a batch size of 128, the GAN produced the following images:

Generated Images

The loss plots show the generator and discriminator loss over the course of training:

Loss Plot
