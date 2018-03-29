# Generative-Adversarial-Network

Keras implementation of Generative Adversarial Networks running on light infrastructure (computer, CPU instance, ...). We provide an opensource implementation of:

- A class DCGAN.GAN to create a Deep Convolutional GAN, based on * Unsupervised representation learning with deep convolutional generative adversarial networks* (Radford et al.).

- A method DCGAN.run_GAN_training that iterates *epochs* RMSprop iterations on the Generative Adversarial Network, on a given dataset. It saves the model and prints a set of generated images frequently to help you monitor training and re-start it through DCGAN.continue_GAN_training if there is a bug/if you need to stop your machine. GAN training can be extremely long!

# Requirements

You just need to have Python 3.6.1 installed and to run:

	git clone https://github.com/samuelpilcer/Generative-Adversarial-Network.git
	cd Generative-Adversarial-Network
	pip install -r requirements.txt

# Run a demo

The following command will give you a short demo on how the library works:

	python examples/run_mnist_demo.py