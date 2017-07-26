import argparse
import numpy as np
import cPickle

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--outDir', default='../Experiments/dcgan', type=str)
	parser.add_argument('--maxEpochs', default=10, type=int) 
	parser.add_argument('--lr', default=1e-3, type=float)  #learning rate
	parser.add_argument('--batchSize', default=64, type=int)
	args = parser.parse_args()
	return args

	#function to MNIST load data
def load_MNIST(opts):
	f = gzip.open('../InData/mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	return train[0].reshape(-1,1,28,28).astype(floatX), train[1], \
	test[0].reshape(-1,1,28,28).astype(floatX), test[1], \
	val[0].astype(-1,1,28,28).astype(floatX), val[1]


def load_CelebA():
	print('Loading celebA data...')
	f = gzip.open('../InData/celeba.npy.gz', 'rb')
	data=cPickle.load(f,mmap_mode='r').astype(floatX)
    print 'CelebA: shape:', np.shape(data), 'min:', data.min(), ,'max:' data.max()
    return data


