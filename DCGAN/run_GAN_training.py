import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

import sys
sys.path.append("../")
import os
from DCGAN.GAN import DCGAN
from cv2 import cv2


class RUN_DCGAN(object):
    def __init__(self, file_folder="mnist", img_rows=28, img_cols=28):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = 1

        if file_folder=="mnist":
            self.x_train = input_data.read_data_sets("mnist",one_hot=True).train.images
            self.x_train = self.x_train.reshape(-1, self.img_rows,self.img_cols, 1).astype(np.float32)

        else:
            self.x_train=[]
            print("Loading images from the folder: "+str(file_folder))
            for i in os.listdir(str(file_folder)):
                self.x_train.append(cv2.resize(cv2.imread(str(file_folder)+i), (self.img_rows, self.img_cols)))
            self.x_train = np.array(self.x_train).reshape(-1, self.img_rows,self.img_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN(img_rows=self.img_rows, img_cols=self.img_cols)
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0, save_folder="", save_model_folder=""):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))
                    self.save_model(save_model_folder, (i+1)/save_interval)

    def save_model(self, save_model_folder, id):
        self.discriminator.save_weights(save_model_folder+"/model_dis_"+str(id)+".h5")
        self.adversarial.save_weights(save_model_folder+"/model_adv_"+str(id)+".h5")
        self.generator.save_weights(save_model_folder+"/model_gen_"+str(id)+".h5")

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0, save_folder=""):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__=="__main__":
    mnist_dcgan = RUN_DCGAN(file_folder="../../wikiart/Action_painting/", img_rows=28, img_cols=28)
    mnist_dcgan.train(train_steps=100, batch_size=256, save_interval=500)
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)