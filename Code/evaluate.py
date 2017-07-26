# functions in here are used for evluatin the models
import numpy as np
from skimage.io import imsave


def montage_RGB(samples, rows, cols):
	"""

	Parameters
	-----------

	samples: np-ndarray
		image samples in a tensor with shape (N,3,width,heigh)

	rows: int
		no of rows of samples

	cols: int
		no of cols of samples
	"""
	assert samples.shape[0] > rows*cols, "less samples than row * cols"

	montage=[]
	for i in range(rows):
		col = samples[i*cols:(i+1)*cols].transpose(1,2,0)
		col = np.hstack(col)
		print np.shape(col)
		montage.append(col)
	montage = np.vstack(montage)
	print np.shape(montage)
	
	return montage



#generations (GAN, AAE)
def eval_gen(sample_fn, nz, outDir, mean=0.0, std=1.0, rows=5, cols=5):
	"""

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


	"""
	noSamples=rows*cols
	Z = np.random.normal(loc=mean, scale=std, size=(noSamples,nz)).astype(floatX) 
	print sample_fn
	X = sample_fn(Z)

	montage = montage_RGB(X)
	imsave((os.path,join(outDir,'montage.png'),montage))

	return montage

#recosntructions (AAE)

#