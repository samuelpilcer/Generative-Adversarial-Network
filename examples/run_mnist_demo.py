import sys
sys.path.append("../")
from DCGAN.GAN import DCGAN
from DCGAN.run_GAN_training import RUN_DCGAN


if __name__=="__main__":
    mnist_dcgan = RUN_DCGAN()
    mnist_dcgan.train(train_steps=1000, batch_size=256, save_interval=500)
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)