from __future__ import print_function, division

#from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

from PIL import Image
import os
import os.path


SRCIMGPATH = r"/media/ksteinfe/DATA/RRSYNC/temp/house_gan_training_14"
DSTIMGPATH = r"/home/ksteinfe/Desktop/TEMP"
img_path_plates = os.path.join(DSTIMGPATH, "plates")
img_path_tiles = os.path.join(DSTIMGPATH, "tiles")

if not os.path.exists(SRCIMGPATH):
    print("Could not find src path: {}".format(SRCIMGPATH))
    exit()
if not os.path.exists(DSTIMGPATH):
    print("Could not find dest path: {}".format(DSTIMGPATH))
    exit()

if not os.path.exists(img_path_plates): os.makedirs(img_path_plates)
if not os.path.exists(img_path_tiles): os.makedirs(img_path_tiles)

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_shape=(self.latent_dim,)))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)



    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        #(X_train, _), (_, _) = mnist.load_data()

        print("====================")
        X_train = self.get_images()
        print(X_train.shape)


        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Sample generator input
            noise = np.random.normal(0, 1, (batch_size, 100))

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)


    def get_images(self):

        imgpaths = [os.path.join(SRCIMGPATH, f) for f in os.listdir(SRCIMGPATH) if os.path.isfile(os.path.join(SRCIMGPATH, f))]
        imgpaths = list(filter(lambda p: p.endswith("jpg"),imgpaths))
        ret = []
        for imgpath in imgpaths:
            img = Image.open(imgpath)
            img = img.convert('L') #makes it greyscale
            matrix = np.array(img)
            ret.append(matrix)

        ret = np.array(ret)
        print(ret.shape)
        return ret

    def save_imgs(self, epoch):
        r, c = 10, 10 # number of images to save out (rows and columns)
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # cut PIL images to save
        pil_imgs = [ self.generated_img_to_PIL_img(nparr) for nparr in gen_imgs ]

        for n, img in enumerate(pil_imgs):
            img.save(os.path.join(img_path_tiles,'{:04d}-{:03d}.png'.format(epoch,n)))

        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(img_path_plates,"house_%d.png" % epoch))
        plt.close()


    def generated_img_to_PIL_img(self, nparr):
        #print(nparr.shape)
        #pxls = [ [px[0] for px in row] for row in nparr ] # pixel values are arrays of a single number for some reason
        pxls = np.squeeze(nparr, axis=2)
        return Image.fromarray(np.uint8(pxls * 255) , 'L')

if __name__ == '__main__':
    dcgan = DCGAN()
    #dcgan.train(epochs=4000, batch_size=32, save_interval=50)
    dcgan.train(epochs=100000, batch_size=32, save_interval=50)
