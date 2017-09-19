"""
Bidirectional Generative Adversarial Networks adapted for the CelebA and MNIST database.
Trains three adversarial networks: generator, encoder & discriminator 
to produce fake but real-looking images and to learn their low dimensional representation.
"""

from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, flatten, reshape, batch_norm, Upscale2DLayer
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import sigmoid
from lasagne.layers import get_output, get_all_params, get_output_shape, get_all_layers
from lasagne.objectives import binary_crossentropy as bce
from lasagne.updates import adam, sgd

import numpy as np
import theano
from theano import tensor as T
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.io import imsave

from functions import get_args, load_CelebA, load_MNIST, print_layers
from nets import get_bigan_gen_mnist, get_bigan_dis_mnist, get_bigan_enc_mnist, get_bigan_gen_celebA, get_bigan_dis_celebA, get_bigan_enc_celebA
from evaluate import eval_gen

import os

floatX=theano.config.floatX


def build_net(nz=100):
	"""
	Get the structure of the generator (gen) and the discriminator (dis) adapted for CelebA or for MNIST
	opts.celeba : argument defined by the user to apply the dcgan on CelebA database
	opts.mnist : argument defined by the user to apply the dcgan on MNIST database

	Parameters
	----------
	nz: int
		Size of the latent code

	Returns
	----------
	gen: class 'layers' or turple
		Structure of the generator. Takes in input the latent code and return an image.
	enc: class 'layers' or turple
		Structure of the encoder. Takes in input an image and return the latent code.
	dis: class 'layers' or turple
		Structure of the discriminator. Takes in input an image and returns a probability.
	"""
	if opts.celeba:
		gen = get_bigan_gen_celebA(nz = nz)
		enc = get_bigan_enc_celebA(nz = nz)
		dis = get_bigan_dis_celebA(nz = nz)

	if opts.mnist:
		gen = get_bigan_gen_mnist(nz = nz)
		enc = get_bigan_enc_mnist(nz = nz)
		dis = get_bigan_dis_mnist(nz = nz)

	return gen, enc, dis


def prep_train(lr=0.0002, nz=100):

	"""
	Update the parameters of the network using gradient descent w.r.t. the loss function.
	The gradient descent is computed using Adam optimiser (adam).

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
		test_fns['samples'] takes un input z (theano.tensor.matrix)


	G: class 'layer' or turple
		Structure of the generator. Takes in input the latent code and return an image.

	D: class 'layer' or turple
		Structure of the discriminator. Takes in input an image and returns a probability.

	"""

	G,E,D=build_net(nz=nz)

	### preparing symbolic variables for each network
	x_enc = T.tensor4('x_enc')
	x = T.tensor4('x')
	z_gen = T.matrix('z_gen')  
	z = T.matrix('z') 
	target_enc = T.matrix('target_enc')
	target_gen = T.matrix('target_gen')
	target_dis = T.matrix('target_dis')

	### compute output of the network
	z_enc = get_output(E, inputs={x_enc:x_enc})
	x_gen = get_output(G, inputs={z_gen:z_gen})
	D_enc = get_output(D,inputs={x_dis:x_enc,z_dis:z_enc})
	D_gen = get_output(D,inputs={x_dis:x_gen,z_dis:z_gen})
	D_dis = get_output(D,inputs={x_dis:x,z_dis:z})

	### test functions
	samples_gen=get_output(G,z_gen,deterministic=True)
	samples_z=get_output(E,x_enc,deterministic=True)
	samples_enc = get_output(G,samples_z, deterministic=True)

	### loss function
	J_E = bce(D_enc,target_enc).mean()
	J_G = bce(D_gen,target_gen).mean()
	J_D = bce(D_dis,target).mean()

	### preparing the update
	params_enc=get_all_params(E, trainable=True) 
	params_gen=get_all_params(G, trainable=True)
	params_dis=get_all_params(D, trainable=True) 

	grad_enc=T.grad(J_E,params_enc)
	grad_dis=T.grad(J_D,params_dis)
	grad_gen=T.grad(J_G,params_gen)

	update_E = optimizer(grad_enc,params_enc,learning_rate = learning_rate,beta1=beta1)
	update_D = optimizer(grad_dis,params_dis,learning_rate = learning_rate,beta1=beta1)
	update_G = optimizer(grad_gen,params_gen,learning_rate = learning_rate,beta1=beta1)

	### train the theano function
	train_fns={}
	train_fns['enc'] =  theano.function(inputs=[x_enc,target_enc], outputs=[J_E], updates=update_E)
	train_fns['gen'] =  theano.function(inputs=[z_gen,target_gen], outputs=[J_G], updates=update_G)
	train_fns['dis'] =  theano.function(inputs=[z,x,target_dis], outputs=[J_D], updates=update_D)

	### test the generator
	test_fns={}
	test_fns['sample_gen']=theano.function(inputs=[z_gen],outputs=samples_gen)
	test_fns['sample_enc']=theano.function(inputs=[x_enc],outputs=samples_enc)


	return train_fns, test_fns, G, E, D

def predict_generator(z):
  return get_output(generator, inputs={z_gen:z}).eval()

def predict_encoder(x):
  return get_output(encoder, inputs={x_enc:x}).eval()



def train(nz=100, lr=0.0002, batchSize=64, epoch=10, outDir='../Experiment/bigan'):

	"""
	Trains the adversarial networks by batch for a certain number of epochs.
	It prints the costs of the generator and the discriminator for every batch. 
	It saves the costs obtained w.r.t the number of batches in a .png image.

	The lower is the cost, the better is the network. 
	Best results are obtained for cost_gen = cost_enc = ln(4) and cost_dis = ln(0.5) [1]

	Note that the generator and the encoder are colluding in order to fool the discriminator [2,3]


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

	E: class 'layer' or turple
		Structure of the encoder

	D: class 'layer' or turple
		Structure of the discriminator


	Reference
	---------
	..  [1] "Generative Adversarial Networks." Goodfellow et al. ArXiv 2014
	..  [2] "Adversarial Feature Learning." Donahue et al. ArXiv 2016
	..  [3] "Adversarially Learned Inference." Dumoulin et al. ArXiv 2016
	"""
	# load the images for training
	if opts.celeba : 
		xTrain = load_CelebA()
	if opts.mnist : 
		xTrain,_,_,_,_,_ = load_MNIST()
	print 'Images for training -- shape:{}, min:{}, max:{} '.format(np.shape(xTrain), np.min(xTrain), np.max(xTrain))

	train_fns, test_fns, G, E, D = prep_train(nz=nz, lr=lr)

	sn,sc,sx,sy=np.shape(xTrain)
	batches=int(np.floor(float(sn)/batchSize))

	#keep training info
	g_cost=[]
	e_cost=[]
	d_cost=[]

	timer=time.time()
	#Train D (outerloop)
	print 'epoch \t batch \t cost G \t cost E \t cost D \t time (s)'
	for e in range(epoch):
		#Do for all batches
		for b in range(batches):

			Z = np.random.normal(loc=0.0, scale=1.0, size=(sn,nz)).astype(floatX) 
			imgs_ = predict_generator(Z[b*batchSize:(b+1)*batchSize]).astype('float32')
			imgs = xTrain[b*batchSize:(b+1)*batchSize].astype('float32')
			Z_ = predict_encoder(imgs).astype('float32')
      	valid = np.ones((batches, 1)).astype('float32')
      	fake = np.zeros((batches, 1)).astype('float32')

      	cost_D_real = train_fns['dis'](z_, imgs, valid)
      	cost_D_fake = train_fns['dis'](z, imgs_, fake)
      	cost_D = 0.5 * np.add(cost_D_real, cost_D_fake)

      	cost_G = train_fns['gen'](Z,valid)    
      	cost_E = train_fns['enc'](imgs,fake)
			
			# print e,'\t',b,'\t',cost_G,'\t', cost_E,'\t', cost_D,'\t', time.time()-timer
			# timer=time.time()
			# g_cost.append(cost_G)
			# d_cost.append(cost_D)
			# e_cost.append(cost_E)


	#save plot of the cost
	plt.plot(range(batches*epoch),g_cost, label="G")
	plt.plot(range(batches*epoch),d_cost, label="D")
	plt.plot(range(batches*epoch),e_cost, label="E")
	plt.legend()
	plt.xlabel('epoch')
	plt.savefig(os.path.join(outDir,'cost_regular.png'))

	return train_fns, test_fns, G, E, D


if __name__ == '__main__':
	opts = get_args()

	#print the layers and their shape of the generator and discriminator
	if opts.printLayers:
		G,E,D=build_net(nz=opts.nz)
		print_layers(G, nn_prefix='generator')
		print_layers(E, nn_prefix='encoder')
		print_layers(D, nn_prefix='discriminator')

	#train the adversarial network 
	train_fns, test_fns, G, E, D = train(nz=opts.nz, lr=opts.lr, batchSize=opts.batchSize, epoch=opts.maxEpochs \
		, outDir=opts.outDir)

	# test the model by printing generated images
	montage_gen = eval_gen(test_fns['sample_gen'], opts.nz, opts.outDir)






