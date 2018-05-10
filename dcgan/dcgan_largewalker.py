from __future__ import print_function, division

# from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam

import numpy as np

from PIL import Image
import os
import os.path

SAVEDMODELNAME = "saved_generator.h5"
SAVEDMODELPATH = r"C:\Users\kstei\Desktop\houseGAN r05 saved model"
DSTIMGPATH = r"C:\Users\kstei\Desktop\TEMP"

#STEPSIZE = 0.2
STEPCOUNT = 1440

img_path_walk = os.path.join(DSTIMGPATH, "walk")

if not os.path.exists(SAVEDMODELPATH):
    print("Could not find src path: {}".format(SAVEDMODELPATH))
    exit()
if not os.path.exists(DSTIMGPATH):
    print("Could not find dest path: {}".format(DSTIMGPATH))
    exit()

if not os.path.exists(img_path_walk): os.makedirs(img_path_walk)

def main():
    pt_orgn =  np.random.normal(0, 1, (1, 100))
    pt_dest =  np.random.normal(0, 1, (1, 100))
    generator = load_model(os.path.join(SAVEDMODELPATH,SAVEDMODELNAME))
    for n in range(STEPCOUNT):
        print("sampling model at {}".format(n))
        gen_img = sample_generator(generator, pt_orgn, pt_dest, n/float(STEPCOUNT))
        save_img(gen_img, "test_{:03d}".format(n))

def sample_generator(generator, pt_o, pt_d, t):
    #noise = pt + (np.random.normal(0, 1, (1, 100)) * STEPSIZE)
    noise = ((pt_d - pt_o) * t) + pt_o
    gen_img = generator.predict(noise)
    #print(gen_img.shape)
    return gen_img

def save_img(nparr, name):
    img = generated_img_to_PIL_img(nparr)
    img.save(os.path.join(img_path_walk, '{}.png'.format(name)))

def generated_img_to_PIL_img(nparr):
    # print(nparr.shape)
    # pxls = [ [px[0] for px in row] for row in nparr ] # pixel values are arrays of a single number for some reason
    #pxls = np.squeeze(nparr, axis=2)
    pxls = np.reshape( nparr, (100,100) )
    return Image.fromarray(np.uint8(pxls * 255), 'L')
    
    
if __name__ == '__main__':
    main()