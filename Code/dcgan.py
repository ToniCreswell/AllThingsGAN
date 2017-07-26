"""
Deep Convolutional Generative Adversarial Networks adapted for the CelebA database.
Trains two adversarial networks (generator & discriminator) to produce fake but real-looking images.

"""

from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, \
flatten, reshape, batch_norm, Upscale2DLayer
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import sigmoid
from lasagne.layers import get_output, get_all_params, get_output_shape, get_all_layers
from lasagne.objectives import binary_crossentropy as bce
from lasagne.updates import adam

import numpy as np
import theano
from theano import tensor as T
import time
from matplotlib import pyplot as plt 

from skimage.io import imsave

from functions import get_args, load_CelebA, print_layers
from nets import get_gen_celebA, get_dis_celebA
from evaluate import eval_gen

import os

floatX=theano.config.floatX


def build_net(nz=100):
	"""
	Get the structure of the generator (gen) and the discriminator (dis) adapted for CelebA  

	Parameters
	----------
	nz: int
		Size of the latent code

	Returns
	----------
	gen: class 'layers' or turple
		Structure of the generator. Takes in input the latent code and return an image.

	dis: class 'layers' or turple
		Structure of the discriminator. Takes in input an image and returns a probability.
	"""
	gen = get_gen_celebA(nz=nz)
	dis = get_dis_celebA(nz=nz)

	return gen, dis


def prep_train(lr=0.0002, nz=100):

	"""
	Update the parameters of the network using gradient descent w.r.t. the loss function.
	The gradient descent is computed using Adam optimiser (adam).
	The loss function is obtained using binary cross entropy (bce).

	Parameters
	----------
	lr: float
		Learning rate - might be smaller than 2e-3

	nz: int
		Size of the latent space code

	Returns
	----------
	train_fns: theano function
		Returns the cost of the gen. (J_G) and the dis. (J_D) w.r.t. their updates for training
		train_fns['dis'] takes in input x (theano.tensor.matrix) and z (theano.tensor.tensor4)
		train_fns['gen'] takes in input z (theano.tensor.tensor4)

	test_fns: theano function
		Function returning images generated for testing
		train_fns['samples'] takes un input z (theano.tensor.matrix)


	G: class 'layer' or turple
		Structure of the generator. Takes in input the latent code and return an image.

	D: class 'layer' or turple
		Structure of the discriminator. Takes in input an image and returns a probability.

	"""
	G,D=build_net(nz=nz)

	x = T.tensor4('x')
	z = T.matrix('z')

	#Get outputs G(z), D(G(z))) and D(x)
	G_z=get_output(G,z)
	D_G_z=get_output(D,G_z)
	D_x=get_output(D,x)

	# test samples
	samples=get_output(G,z,deterministic=True)

	#Get parameters of G and D
	params_d=get_all_params(D, trainable=True)
	params_g=get_all_params(G, trainable=True)

	#Calc loss and updates
	J_D = bce(D_x,T.ones_like(D_x)).mean() + bce(D_G_z,T.zeros_like(D_G_z)).mean() #mean over a batch
	J_G = bce(D_G_z, T.ones_like(D_G_z)).mean() #mean over a batch ("stronger gradients in early training")

	grad_d=T.grad(J_D,params_d)
	grad_g=T.grad(J_G,params_g)

	update_D = adam(grad_d,params_d, learning_rate=lr)
	update_G = adam(grad_g,params_g, learning_rate=lr)

	#theano train functions
	train_fns={}
	train_fns['gen']=theano.function(inputs=[z], outputs=J_G, updates=update_G)
	train_fns['dis']=theano.function(inputs=[x,z], outputs=J_D, updates=update_D)

	#theano test functions
	test_fns={}
	test_fns['sample']=theano.function(inputs=[z],outputs=samples)

	return train_fns, test_fns, G, D

def train(nz=100, lr=0.0002, batchSize=64, epoch=10, outDir='../Experiment/dcgan'):

	"""
	Trains the adversarial networks by batch for a certain number of epochs.
	It prints the costs of the generator and the discriminator for every batch. 
	It saves the costs obtained w.r.t the number of batches in a .png image.

	The lower is the cost, the better is the network. 
	Best results are obtained for cost_gen = ln(4) and cost_dis = ln(0.5) [1]


	Parameters
	----------
	nz: int
		Size of the latent code

	lr: float
		Learning rate

	batchSize: int
		Size of one batch 

	epoch: int
		Number of epochs - one epoch would go through all the dataset

	outDir: str
		Path directory for the results to be saved


	Returns
	---------
	train_fns: theano.function
		Return the cost of the generator and the discriminator for training

	test_fns: theano.function
		Return the images generated by the generator for testing

	G: class 'layer' or turple
		Structure of the generator

	D: class 'layer' or turple
		Structure of the discriminator


	Reference
	---------
	..  [1] "Generative Adversarial Networks." Goodfellow et al. ArXiv 2014
	"""

	xTrain= load_CelebA()
	train_fns, test_fns, G, D = prep_train(nz=nz, lr=lr)

	sn,sc,sx,sy=np.shape(xTrain)
	batches=int(np.floor(float(sn)/batchSize))

	#keep training info
	g_cost=[]
	d_cost=[]

	timer=time.time()
	#Train D (outerloop)
	print 'epoch \t batch \t cost G \t\t cost D \t\t time (s)'
	for e in range(epoch):
		#Do for all batches
		for b in range(batches):
			for k in range(1):
				Z = np.random.normal(loc=0.0, scale=1.0, size=(sn,nz)).astype(floatX) #Normal prior, P(Z)
				#Go through one batch
				cost_D=train_fns['dis'](xTrain[b*batchSize:(b+1)*batchSize],Z[b*batchSize:(b+1)*batchSize])
			#Train G (inerloop)
			#Go through one batch
			Z = np.random.normal(loc=0.0, scale=1.0, size=(sn,nz)).astype(floatX) 
			cost_G=train_fns['gen'](Z[b*batchSize:(b+1)*batchSize])
			print e,'\t',b,'\t',cost_G,'\t', cost_D,'\t', time.time()-timer
			timer=time.time()
			g_cost.append(cost_G)
			d_cost.append(cost_D)


	#save plot of the cost
	plt.plot(range(batches*epoch),g_cost, label="G")
	plt.plot(range(batches*epoch),d_cost, label="D")
	plt.legend()
	plt.xlabel('epoch')
	print outDir, '   ', os.path.join(outDir,'cost_regular.png')
	plt.savefig(os.path.join(outDir,'cost_regular.png'))

	return train_fns, test_fns, G, D


if __name__ == '__main__':
	opts = get_args()
	print 'outDir:', opts.outDir

	#print the layers out with sizes
	if opts.printLayers:
		G,D=build_net(nz=opts.nz)
		print_layers(G, nn_prefix='generator')
		print_layers(D, nn_prefix='discriminator')

	train_fns, test_fns, G, D = train(nz=opts.nz, lr=opts.lr, batchSize=opts.batchSize, epoch=opts.maxEpochs \
		, outDir=opts.outDir)

	print test_fns['sample']
	montage = eval_gen(test_fns['sample'], opts.nz, opts.outDir)




# G_Z=test(G).eval()

# #see if the output images look good:
# imsave('testOutputs/text.png',G_Z[0].transpose(1,2,0))
# sn,sc,sx,sy=np.shape(x_train)
# montage=np.ones(shape=(10*sx,10*sy,3))

# n=0
# x=0
# y=0
# for i in range(10):
#     for j in range(10):
#         im=G_Z[n,:,:,:].swapaxes(0, 2)
#         n+=1
#         montage[x:x+sx,y:y+sy,:]=im
#         x+=sx
#     x=0
#     y+=sy
# print 'montage:',np.shape(montage)
# imsave('testOutputs/montage.png',montage)




