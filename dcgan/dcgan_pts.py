from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np

import os
import os.path

SRCIMGPATH = r"C:\Users\kstei\Desktop\HousePts GAN v01a"
DSTIMGPATH = r"C:\Users\kstei\Desktop\TEMP"
SAMPLESPERINTERVAL = 10

if not os.path.exists(SRCIMGPATH):
    print("Could not find src path: {}".format(SRCIMGPATH))
    exit()
if not os.path.exists(DSTIMGPATH):
    print("Could not find dest path: {}".format(DSTIMGPATH))
    exit()

class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
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

        model.add(Dense(128 * 8 * 8, activation="relu", input_shape=(self.latent_dim,)))
        model.add(Reshape((8, 8, 128)))
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
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
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

        print("====================")
        X_train = self.get_images()
        print(X_train.shape) # make sure this guy is correct

        # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

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
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
        self.generator.save(os.path.join(DSTIMGPATH,'pts_saved_generator.h5'))
        self.discriminator.save(os.path.join(DSTIMGPATH,'pts_saved_discriminator.h5'))

    def get_images(self):
        imgpaths = [os.path.join(SRCIMGPATH, f) for f in os.listdir(SRCIMGPATH) if
                    os.path.isfile(os.path.join(SRCIMGPATH, f))]
        ret = []
        for imgpath in imgpaths:
            try:
                lhs = []
                with open(imgpath) as f:
                    for x in f.readlines():
                        lhs.append([float(y) for y in x.split(',')])

                new_data = np.array(lhs)
                #print(new_data.shape)
                if (len(lhs)>0):
                    new_data = np.reshape(new_data, (32,32,3))
                    ret.append(new_data)
            except Exception as e:
                print("could not load xyz: {}\n{}".format(imgpath, e))

        ret = np.array(ret)
        print('-------------------------')
        print('dataset shape')
        print(ret.shape)
        print('-------------------------')
        return ret

    def save_imgs(self, epoch):
        noise = np.random.normal( 0, 1, (SAMPLESPERINTERVAL,100) )
        gen_imgs = self.generator.predict(noise)
        print (gen_imgs.shape)

        file_count = 0
        for x in gen_imgs:
            csv = ""
            for row in x.reshape(1024,3):
                csv += """{}, {}, {} \n""".format(*row)
            with open(os.path.join(DSTIMGPATH,'house_{:05d}_{:03d}.xyz'.format(epoch, file_count)), 'w') as f:
                f.write(csv)
            file_count += 1




if __name__ == '__main__':
    dcgan = DCGAN()
    #dcgan.train(epochs=1, batch_size=32, save_interval=50)
    #dcgan.train(epochs=4000, batch_size=32, save_interval=50)
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
