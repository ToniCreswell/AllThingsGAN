"""
All current networks to be used for the Adversarial Models
"""

################################## dcgan Nets ##################################

"""
Generative network adapted for the CelebA/MNIST database

The generative network is composed of deconvolution layers. 
It takes in input a variable from the latent space (matrix).
It returns an image (4D tensor). 

Parameters
----------
nz: int 
	Number of variables in the latent space

Returns
---------
gen: class 'Layer' instance or a tuple
	Return the generative network 


*							*							*


Discriminative network adapted for the CelebA/MNIST database

The discriminative network is composed of deconvolution layers. 
It takes in input an image (4D tensor).
It returns a probability (float from 0 to 1). 

Parameters
----------
nz: int 
	Number of variables in the latent space

Returns
--------
dis: class 'layer' instance or tuple
	Return the discriminative network 

"""

from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, flatten, \
reshape, batch_norm, Upscale2DLayer, dropout, concat, Pool2DLayer
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import sigmoid, softmax


def get_gen_celebA(nz=100):

	gen = InputLayer(shape=(None,nz))
	gen = DenseLayer(incoming=gen, num_units=1024*4*4)
	gen = reshape(incoming=gen, shape=(-1,1024,4,4))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=512, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=256, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=128, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = Deconv2DLayer(incoming=gen, num_filters=3, filter_size=4, stride=2, nonlinearity=sigmoid, crop=1)
	return gen

def get_dis_celebA(nz=100):

	dis = InputLayer(shape=(None,3,64,64))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=128, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=256, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=512, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=1024, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 
	dis = reshape(incoming=dis, shape=(-1,1024*4*4))
	dis = DenseLayer(incoming=dis, num_units=1, nonlinearity=sigmoid)
	return dis

def get_gen_mnist(nz=100):

	gen = InputLayer(shape=(None,nz))
	gen = DenseLayer(incoming=gen, num_units=512*2*2)
	gen = reshape(incoming=gen, shape=(-1,512,2,2))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=256, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=128, filter_size=3, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=64, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = Deconv2DLayer(incoming=gen, num_filters=1, filter_size=4, stride=2, nonlinearity=sigmoid, crop=1)
	return gen

def get_dis_mnist(nz=100):

	dis = InputLayer(shape=(None,1,28,28))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=64, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=128, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=256, filter_size=3,stride=2, nonlinearity=lrelu(0.2),pad=1))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=512, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = reshape(incoming=dis, shape=(-1,512*2*2))
	dis = DenseLayer(incoming=dis, num_units=1, nonlinearity=sigmoid)
	return dis


################################### wgan Nets ##################################

	"""
	generative and discriminative networks are the globally the same than the ones used for dcgan. 
	The only exception is that, because the loss function of a WGAN is not a bce, the last layer shouldn't have any nonlinearity.
	"""

def get_wgen_celebA(nz=100):

	gen = InputLayer(shape=(None,nz))
	gen = DenseLayer(incoming=gen, num_units=1024*4*4)
	gen = reshape(incoming=gen, shape=(-1,1024,4,4))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=512, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=256, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=128, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = Deconv2DLayer(incoming=gen, num_filters=3, filter_size=4, stride=2, nonlinearity=None, crop=1)
	return gen

def get_wdis_celebA(nz=100):

	dis = InputLayer(shape=(None,3,64,64))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=128, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=256, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=512, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=1024, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 
	dis = reshape(incoming=dis, shape=(-1,1024*4*4))
	dis = DenseLayer(incoming=dis, num_units=1, nonlinearity=None)
	return dis

def get_wgen_mnist(nz=100):

	gen = InputLayer(shape=(None,nz))
	gen = DenseLayer(incoming=gen, num_units=512*2*2)
	gen = reshape(incoming=gen, shape=(-1,512,2,2))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=256, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=128, filter_size=3, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=64, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = Deconv2DLayer(incoming=gen, num_filters=1, filter_size=4, stride=2, nonlinearity=None, crop=1)
	return gen

def get_wdis_mnist(nz=100):

	dis = InputLayer(shape=(None,1,28,28))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=64, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=128, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=256, filter_size=3,stride=2, nonlinearity=lrelu(0.2),pad=1))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=512, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = reshape(incoming=dis, shape=(-1,512*2*2))
	dis = DenseLayer(incoming=dis, num_units=1, nonlinearity=None)
	return dis

################################### bidirectional gan Nets ############################

	"""
	The generative network is the same than the one used for dcgan. 
	an encoder network is added, and is used to collude with the generetor in order to fool the discriminator.
	The discriminator concatenates two inputs: the image (resized) and its lower representation.
	Because the network is massive, dropout layers have been added. 
	"""

def get_bigan_gen_celebA(nz=100):
	# latent code --> image
    z_gen = InputLayer(shape=(None,nz))
    gen = batch_norm(DenseLayer(incoming=z_gen, num_units=1024*4*4, nonlinearity = lrelu(0.2)))
    gen = batch_norm(reshape(incoming=gen, shape=(-1,1024,4,4)))
    gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=512, filter_size=4, stride=2, nonlinearity=lrelu(0.2), crop=1))
    gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=256, filter_size=4, stride=2, nonlinearity=lrelu(0.2), crop=1))
    gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=128, filter_size=4, stride=2, nonlinearity=lrelu(0.2), crop=1))
    gen = Deconv2DLayer(incoming=gen, num_filters=3, filter_size=4, stride=2, crop=1, nonlinearity=sigmoid)
    return z_gen, gen

def get_bigan_enc_celebA(nz=100):
	# image --> latent code
    x_enc = InputLayer(shape=(None,3,64,64))
    enc = batch_norm(Conv2DLayer(incoming = x_enc, num_filters = 128, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2)))
    enc = batch_norm(Pool2DLayer(incoming = enc, pool_size = 2, stride=None, pad=(0, 0), ignore_border=True, mode='average_exc_pad'))
    enc = batch_norm(Conv2DLayer(incoming = enc, num_filters = 256, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2)))
    enc = batch_norm(Pool2DLayer(incoming = enc, pool_size = 2, stride=None, pad=(0, 0), ignore_border=True, mode='average_exc_pad'))
    enc = Conv2DLayer(incoming = enc, num_filters = 512, filter_size = 5, stride = 2, pad = 2, nonlinearity = sigmoid)
    enc = flatten(incoming = enc, outdim = 2)
    enc = DenseLayer(incoming=enc, num_units=nz, nonlinearity= None)
    return x_enc, enc

def get_bigan_dis_celebA(nz=100):
	# (latent code, image) --> probability of being fake/true
    z_dis = InputLayer(shape=(None,nz))
    x_dis = InputLayer(shape=(None,3,64,64))
    x_dis_int = batch_norm(Conv2DLayer(incoming = x_dis, num_filters=32, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2)))
    x_dis_int = dropout(incoming = batch_norm(Conv2DLayer(incoming = x_dis_int, num_filters=64, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2))), p=0.2)
    x_dis_int = dropout(incoming = batch_norm(Conv2DLayer(incoming = x_dis_int, num_filters=128, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2))), p=0.2)
    x_dis_int = dropout(incoming = batch_norm(Conv2DLayer(incoming = x_dis_int, num_filters=256, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2))), p=0.2)
    x_dis_int = batch_norm(flatten(incoming = x_dis_int, outdim = 2))
    x_dis_int = batch_norm(DenseLayer(incoming = x_dis_int, num_units = nz, nonlinearity = None))    
    dis = concat(incomings = [z_dis, flatten(incoming = x_dis_int, outdim = 2)], axis=-1, cropping=None)
    dis = dropout(incoming = DenseLayer(incoming = dis, num_units = 512, nonlinearity = lrelu(0.2)),p=0.25)
    dis = dropout(incoming = DenseLayer(incoming = dis, num_units = 512, nonlinearity = lrelu(0.2)),p=0.25)
    dis = DenseLayer(incoming = dis, num_units = 1, nonlinearity = sigmoid)
    return z_dis, x_dis, dis



def get_bigan_gen_mnist(nz=100):
	# latent code --> image
    z_gen = InputLayer(shape=(None,nz))
    gen = batch_norm(DenseLayer(incoming=z_gen, num_units=1024*2*2, nonlinearity = lrelu(0.2)))
    gen = batch_norm(reshape(incoming=gen, shape=(-1,1024,2,2)))
    gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=512, filter_size=4, stride=2, nonlinearity=lrelu(0.2), crop=1))
    gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=256, filter_size=4, stride=2, nonlinearity=lrelu(0.2), crop=1))
    gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=128, filter_size=4, stride=2, nonlinearity=lrelu(0.2), crop=1))
    gen = Deconv2DLayer(incoming=gen, num_filters=1, filter_size=4, stride=2, crop=3, nonlinearity=sigmoid)
    return z_gen, gen

def get_bigan_enc_mnist(nz=100):
	# image --> latent code
    x_enc = InputLayer(shape=(None,1,28,28))
    enc = batch_norm(Conv2DLayer(incoming = x_enc, num_filters = 128, filter_size = 4, stride = 2, pad = 3, nonlinearity = lrelu(0.2)))
    enc = batch_norm(Pool2DLayer(incoming = enc, pool_size = 2, stride=None, pad=(0, 0), ignore_border=True, mode='average_exc_pad'))
    enc = batch_norm(Conv2DLayer(incoming = enc, num_filters = 256, filter_size = 4, stride = 2, pad = 1, nonlinearity = lrelu(0.2)))
    enc = batch_norm(Pool2DLayer(incoming = enc, pool_size = 2, stride=None, pad=(0, 0), ignore_border=True, mode='average_exc_pad'))
    enc = Conv2DLayer(incoming = enc, num_filters = 512, filter_size = 4, stride = 2, pad = 1, nonlinearity = sigmoid)
    enc = flatten(incoming = enc, outdim = 2)
    enc = DenseLayer(incoming=enc, num_units=nz, nonlinearity= None)
    return x_enc, enc

def get_bigan_dis_mnist(nz=100):
	# (latent code, image) --> probability of being fake/true
    z_dis = InputLayer(shape=(None,nz))
    x_dis = InputLayer(shape=(None,1,28,28))
    x_dis_int = batch_norm(Conv2DLayer(incoming = x_dis, num_filters=16, filter_size = 4, stride = 2, pad = 3, nonlinearity = lrelu(0.2)))
    x_dis_int = batch_norm(Conv2DLayer(incoming = x_dis_int, num_filters=32, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2)))
    x_dis_int = batch_norm(Conv2DLayer(incoming = x_dis_int, num_filters=64, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2)))
    x_dis_int = batch_norm(Conv2DLayer(incoming = x_dis_int, num_filters=128, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2)))
    x_dis_int = batch_norm(Conv2DLayer(incoming = x_dis_int, num_filters=256, filter_size = 5, stride = 2, pad = 2, nonlinearity = lrelu(0.2)))
    x_dis_int = batch_norm(flatten(incoming = x_dis_int, outdim = 2))
    x_dis_int = batch_norm(DenseLayer(incoming = x_dis_int, num_units = nz, nonlinearity = None))    
    dis = concat(incomings = [z_dis, flatten(incoming = x_dis_int, outdim = 2)], axis=-1, cropping=None)
    dis = dropout(incoming = DenseLayer(incoming = dis, num_units = 512, nonlinearity = lrelu(0.2)),p=0.5)
    dis = DenseLayer(incoming = dis, num_units = 1, nonlinearity = sigmoid)
    return z_dis, x_dis, dis

################################## cAAE Nets ##################################

def get_enc_MNIST():
	# image --> encoding
	enc = InputLayer((None, 28*28))
	enc = DenseLayer(incoming=enc, num_units=1000, nonlinearity=relu)
	enc = DenseLayer(incoming=enc, num_units=1000, nonlinearity=relu)
	return enc


def get_Zenc_MNIST(nz=100):
	#encoding --> latent rep
	Zenc = InputLayer((None, 1000))
	Zenc = DenseLayer(incoming=Zenc, num_units=nz, nonlinearity=None)
	return Zenc


def get_Yenc_MNIST():
	#encoding --> label vector
	Yenc = InputLayer((None, 1000))
	Yenc = DenseLayer(incoming=Yenc, num_units=10, nonlinearity=softmax)  #10 labels
	return Yenc

def get_dec_MNIST(nz=100):
	#[latent , label] --> sample
	dec = InputLayer((None, nz+10))
	dec = DenseLayer(incoming=dec, num_units=1000, nonlinearity=relu)
	dec = DenseLayer(incoming=dec, num_units=1000, nonlinearity=relu)
	dec = DenseLayer(incoming=dec, num_units=28*28, nonlinearity=sigmoid)
	return dec

def get_disZ_MNIST(nz=100):
	# z --> real or fake
	disZ = InputLayer((None, nz))
	disZ = DenseLayer(incoming=disZ, num_units=1000, nonlinearity=relu)
	disZ = DenseLayer(incoming=disZ, num_units=1000, nonlinearity=relu)
	disZ = DenseLayer(incoming=disZ, num_units=1, nonlinearity=sigmoid)
	return disZ

def get_disY_MNIST():
	# y --> real or fake
	disY = InputLayer((None, 10))
	disY = DenseLayer(incoming=disY, num_units=1000, nonlinearity=relu)
	disY = DenseLayer(incoming=disY, num_units=1000, nonlinearity=relu)
	disY = DenseLayer(incoming=disY, num_units=1, nonlinearity=sigmoid)
	return disY











