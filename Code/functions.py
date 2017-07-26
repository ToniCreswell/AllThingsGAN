import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inDir', default='../InData/mnist.pkl.gz', type=str)
	parser.add_argument('--outDir', default='../ExpMOLES', type=str)
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
    data=np.load('../InData/celeba.npy',mmap_mode='r').astype(floatX)
    return data


