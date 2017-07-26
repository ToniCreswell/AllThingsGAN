import argparse
import numpy as np
import cPickle
import gzip, zipfile
import os
from lasagne.layers import get_all_layers, get_output_shape

import theano
floatX = theano.config.floatX

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--outDir', default='../Experiments/dcgan', type=str)
	parser.add_argument('--maxEpochs', default=1, type=int) 
	parser.add_argument('--lr', default=1e-3, type=float)  #learning rate
	parser.add_argument('--batchSize', default=64, type=int)
	parser.add_argument('--printLayers', action= 'store_true')
	parser.add_argument('--nz', default=100, type=int)
	args = parser.parse_args()
	return args

	#function to MNIST load data
def load_MNIST(opts):
	print('Loading MNIST data...')
	f = gzip.open('../InData/mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	return train[0].reshape(-1,1,28,28).astype(floatX), train[1], \
	test[0].reshape(-1,1,28,28).astype(floatX), test[1], \
	val[0].reshape(-1,1,28,28).astype(floatX), val[1]


def load_CelebA():
	print('Loading celebA data...')
	inDir='../InData/celeba.npy'

	if not os.path.exists(inDir):
		zip = zipfile.ZipFile(inDir+'.zip')
		zip.extractall('../InData/')

	data=np.load(inDir,mmap_mode='r').astype(floatX)

	print 'CelebA: shape:', np.shape(data), 'min:', data.min() ,'max:', data.max()
	return data


def print_layers(nn, nn_prefix='nn'):
	nn_layers = get_all_layers(nn)
	l = 0
	for i, layer in enumerate(nn_layers):
		label = '{0}'.format(layer.__class__.__name__)
		print('{:11s} {} : {:25s} - shape : {} '.format(nn_prefix[:10], i,label[:25], get_output_shape(nn_layers[i])))