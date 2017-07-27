"""
Usual functions to be called 
"""

import argparse
import numpy as np
import cPickle
import gzip, zipfile
import os
from lasagne.layers import get_all_layers, get_output_shape

import theano
floatX = theano.config.floatX

def get_args():
	"""
	Get all the arguments defined by the user, e.g. the size of the batch, or the learning rate.


	--outDir: str
		Define the directory path where the results are saved

	--maxEpochs: int
		Number of epochs wanted for the training

	--lr: float
		Learning rate 

	--batchSize: int
		Size of one batch (i.e: number of images in one batch)

	--printLayers: boolean
		Option to print the layers used in the network

	--nz: int
		Number of variables in the latent space 


	Returns
	----------

	args: list
		list of arguments provided by the user calling for the appropriate action 

	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--outDir', required=True, type=str)
	parser.add_argument('--maxEpochs', default=1, type=int) 
	parser.add_argument('--lr', default=1e-3, type=float)  #learning rate
	parser.add_argument('--batchSize', default=64, type=int)
	parser.add_argument('--printLayers', action= 'store_true')
	parser.add_argument('--nz', default=100, type=int)
	args = parser.parse_args()
	return args

	#function to MNIST load data
def load_MNIST(opts):
	"""
	Unzip and load the MNIST dataset [1]


	Parameters
	----------

	opts: list
		list of the arguments previously defined by the function get_args():


	Returns
	--------

	train_im: tensor4D (float)
		images from the MNIST dataset that should be used for training

	train_label: vector (int)
		associated labels used for training

	test_im: tensor 4D (float)
		images from the MNIST dataset that should be used for testing 

	test_label: vector (float)
		associated labels used for testing.

	valid_im: tensor4D (float)
		images from the MNIST dataset that should be used for validation

	valid_label: vector (float)
		associated labels used for validation

	References
	----------
	.. [1] http://deeplearning.net/tutorial/gettingstarted.html

	"""

	print('Loading MNIST data...')
	f = gzip.open('../InData/mnist.pkl.gz', 'rb')
	train, valid, test = cPickle.load(f)
	f.close()

	train_im, train_label, test_im, test_label, valid_im, valid_label \
	= train[0].astype(floatX), train[1], \
	test[0].astype(floatX), test[1], \
	valid[0].astype(floatX), valid[1]

	return train_im, train_label, test_im, test_label, valid_im, valid_label

# args = get_args()
# train_im, train_label, test_im, test_label, valid_im, valid_label = load_MNIST(args)
# print train_label.dtype
# print train_im.dtype

def load_CelebA():
	"""
	Load a reduced dataset from the CelebFaces Attribute dataset [1]

	Returns
	--------

	data: tensor4D
		images from the CelebA dataset 


	References
	---------
	..  [1] http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
	"""

	print('Loading celebA data...')
	inDir='../InData/celeba.npy'

	if not os.path.exists(inDir):
		zip = zipfile.ZipFile(inDir+'.zip')
		zip.extractall('../InData/')

	data=np.load(inDir,mmap_mode='r').astype(floatX)

	print 'CelebA: shape:', np.shape(data), 'min:', data.min() ,'max:', data.max()
	return data


def print_layers(nn, nn_prefix='nn'):
	"""
	Print the layers of the network in the following way: 
	..	name_of_layer, position_of_layer, type_of_layer, shape_of_layer


	Parameters
	----------

	nn: str
		Name refering to the layer

	nn_prefix: str
		Name to be printed

	"""

	nn_layers = get_all_layers(nn)
	l = 0
	for i, layer in enumerate(nn_layers):
		label = '{0}'.format(layer.__class__.__name__)
		print('{:11s} {} : {:25s} - shape : {} '.format(nn_prefix[:10], i,label[:25], get_output_shape(nn_layers[i])))

#Function to apply learned models to data in batches
def apply(model, data, batchSize=64):
	output=[]
	for b in range(np.shape(data)[0]//batchSize):
		out=model(data[b*batchSize:(b+1)*batchSize])
		output.append(out)
	return np.concatenate(output)


