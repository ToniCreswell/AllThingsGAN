"""
All current networks to be used for the Generative Adversarial Models
"""

from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, flatten, reshape, batch_norm, Upscale2DLayer
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import sigmoid


def get_gen_celebA(nz=100):
	"""
	Generative network adapted for the CelebA database

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

	"""
	gen = InputLayer(shape=(None,nz))
	gen = DenseLayer(incoming=gen, num_units=1024*4*4)
	gen = reshape(incoming=gen, shape=(-1,1024,4,4))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=512, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=256, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=128, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = Deconv2DLayer(incoming=gen, num_filters=3, filter_size=4, stride=2, nonlinearity=sigmoid, crop=1)

	return gen

def get_dis_celebA(nz=100):
	"""
	Discriminative network adapted for the CelebA database

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
	dis = InputLayer(shape=(None,3,64,64))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=128, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=256, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=512, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=1024, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 
	dis = reshape(incoming=dis, shape=(-1,1024*4*4))
	dis = DenseLayer(incoming=dis, num_units=1, nonlinearity=sigmoid)
	return dis