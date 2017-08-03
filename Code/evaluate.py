"""
Evaluation of the network
"""

import numpy as np
from skimage.io import imsave
import theano
import os
from functions import load_CelebA, load_MNIST, get_args

floatX = theano.config.floatX

#Images to be tested
opts = get_args()
if opts.celeba:
	samples = load_CelebA()
if opts.mnist:
	_,_,samples,_,_,_ = load_MNIST()

print 'samples shape: ', np.shape(samples)

def montage(samples, rows=5, cols=5):
	"""
	Returns a montage of (rows * cols) RGB-images 

	Parameters
	-----------

	samples: np-ndarray
		image samples in a tensor with shape (N,3,width,heigh)

	rows: int
		number of rows of samples

	cols: int
		number of cols of samples

	Return
	----------

	montage: 4Dtensor
		Returns (rows * cols) of RGB-images
	"""
	assert samples.shape[0] >= rows*cols, "less samples than row * cols"
	montage=[]
	# Images in input are one-channel (mnist)
	if samples.shape[1] == 1:
		samples = samples.transpose(0,2,3,1).squeeze()
		print np.shape(samples)
	# Images in input are rgb-channel (celebA)
	if samples.shape[1] == 3:
		samples = samples.transpose(0,2,3,1)
		print np.shape(samples)

	for i in range(rows):
		col = samples[i*cols:(i+1)*cols]
		col = np.hstack(col)
		montage.append(col)
	montage = np.vstack(montage)
	return montage


def eval_gen(sample_fn, nz, outDir, mean=0.0, std=1.0, rows=5, cols=5):
	"""
	Returns a montage of fake images generated by the generator in order to evaluate the model.
	The images are generated from a latent variable (Z). 
	Z has the shape of a gaussian distribution (np.random.normal) defined by its mean and std.

	Parameters
	-----------
	sample_fn: theano.function
		takes as input a batch of z vectors and outputs a batch of images

	nz: int
		size of z, the latent var passed through the generator

	mean: float
		mean of the guassian distribuiton from which z is drawn

	std: float
		std of the guassian distribution from which z is drawn

	row: int
		no of row of sample images to display

	col: int
		no of coloumns of sample images to display

	Returns
	-----------
	montage: 4Dtensor
		Returns a montage of RGB-images

	"""
	noSamples=rows*cols
	Z = np.random.normal(loc=mean, scale=std, size=(noSamples,nz)).astype(floatX) 
	print sample_fn
	X = sample_fn(Z)

	montage = montage(X, rows=rows, cols=cols)
	imsave(os.path.join(outDir,'montage.png'),montage)
	return montage

