# Generative Adversarial Networks and Autoencoders

This repository aims to present the structure of generative models using **theano** and **lasagne**, and the results obtained using **celebA** and **mnist** datasets. 

You will find the code for: 
* DCGAN: Deep Convolutional Generative Adversarial Networks
* BiGAN: Bidirectional Generative Adversarial Networks
* WGAN: Wasserstein Generative Adversarial Networks
* DAAE: Denoising Adversarial Autoencoders

## Prerequisites
You will need to install a virtual environment containing:
* Theano
* Lasagne
* Python 2.7


More information about lasagne installation can be found at: http://lasagne.readthedocs.io/en/latest/user/installation.html

## Running the code 
The code can be run either for celebA or MNIST. 
For information, the entire celebA dataset can be downloaded at: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

The code can be run in the following way: 
```
python dcgan.py --mnist --printLayers --outDir '~/AllThingsGAN/Experiments/dcgan_mnist/'
```

