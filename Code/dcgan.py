#DCGAN Using Lasagne (for CelebA)
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, flatten, reshape, batch_norm, Upscale2DLayer
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import sigmoid
from lasagne.layers import get_output, get_all_params, get_output_shape
from lasagne.objectives import binary_crossentropy as bce
from lasagne.updates import adam

import numpy as np
import theano
from theano import tensor as T
import time
from matplotlib import pyplot as plt 

from skimage.io import imsave

from functions import get_args, loadData

floatX=theano.config.floatX


def build_net(nz=100):
	# nz = size of latent code
	#N.B. using batch_norm applies bn before non-linearity!
	#Generator networks
	gen = InputLayer(shape=(None,nz))
	gen = DenseLayer(incoming=gen, num_units=1024*4*4)
	gen = reshape(incoming=gen, shape=(-1,1024,4,4))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=512, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=256, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = batch_norm(Deconv2DLayer(incoming=gen, num_filters=128, filter_size=4, stride=2, nonlinearity=relu, crop=1))
	gen = Deconv2DLayer(incoming=gen, num_filters=3, filter_size=4, stride=2, nonlinearity=sigmoid, crop=1)

	dis = InputLayer(shape=(None,3,64,64))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=128, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=256, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2))
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=512, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 
	dis = batch_norm(Conv2DLayer(incoming=dis, num_filters=1024, filter_size=5,stride=2, nonlinearity=lrelu(0.2),pad=2)) 
	dis = reshape(incoming=dis, shape=(-1,1024*4*4))
	dis = DenseLayer(incoming=dis, num_units=1, nonlinearity=sigmoid)

	return gen, dis


def prep_train(alpha=0.0002, nz=100):
	G,D=build_net(nz=nz)

	x = T.tensor4('x')
	z = T.matrix('z')

	#Get outputs G(z), D(G(z))) and D(x)
	G_z=get_output(G,z)
	D_G_z=get_output(D,G_z)
	D_x=get_output(D,x)

	samples=get_output(G,z,deterministic=True)

	#Get parameters of G and D
	params_d=get_all_params(D, trainable=True)
	params_g=get_all_params(G, trainable=True)

	#Calc loss and updates
	J_D = bce(D_x,T.ones_like(D_x)).mean() + bce(D_G_z,T.zeros_like(D_G_z)).mean() #mean over a batch
	J_G = bce(D_G_z, T.ones_like(D_G_z)).mean() #mean over a batch ("stronger gradients in early training")

	grad_d=T.grad(J_D,params_d)
	grad_g=T.grad(J_G,params_g)

	#for p,g in zip(params_d, grad_d):
	#	print p,'+',alpha,'+', g, '\n'

	#update_D = [(param, param - alpha * grad) for param, grad in zip(params_d, grad_d)]
	#update_G = [(param, param - alpha * grad) for param, grad in zip(params_g, grad_g)]
	update_D = adam(grad_d,params_d, learning_rate=alpha)
	update_G = adam(grad_g,params_g, learning_rate=alpha)


	train_G=theano.function(inputs=[z], outputs=J_G, updates=update_G)
	train_D=theano.function(inputs=[x,z], outputs=J_D, updates=update_D)
	test_G=theano.function(inputs=[z],outputs=samples)

	return train_G, train_D, G, D

def train(trainData, nz=100, alpha=0.001, batchSize=64, epoch=10):
	train_G, train_D, G, D = prep_train(nz=nz, alpha=alpha)
	sn,sc,sx,sy=np.shape(trainData)
	print sn,sc,sx,sy
	batches=int(np.floor(float(sn)/batchSize))

	#keep training info
	g_cost=[]
	d_cost=[]

	print 'batches=',batches

	timer=time.time()
	#Train D (outerloop)
	print 'epoch \t batch \t cost G \t\t cost D \t\t time (s)'
	for e in range(epoch):
		#random re-order of data (no doing for now cause slow)
		#Do for all batches
		for b in range(batches):
			for k in range(1):
				Z = np.random.normal(loc=0.0, scale=1.0, size=(sn,nz)).astype(floatX) #Normal prior, P(Z)
				#Go through one batch
				cost_D=train_D(trainData[b*batchSize:(b+1)*batchSize],Z[b*batchSize:(b+1)*batchSize])
			#Train G (inerloop)
			#Go through one batch
			Z = np.random.normal(loc=0.0, scale=1.0, size=(sn,nz)).astype(floatX) 
			cost_G=train_G(Z[b*batchSize:(b+1)*batchSize])
			print e,'\t',b,'\t',cost_G,'\t', cost_D,'\t', time.time()-timer
			timer=time.time()
			g_cost.append(cost_G)
			d_cost.append(cost_D)


		#save plot of the cost
	plt.plot(range(batches*epoch),g_cost, label="G")
	plt.plot(range(batches*epoch),d_cost, label="D")
	plt.legend()
	plt.xlabel('epoch')
	plt.savefig('testOutputs/cost_regular.png')

	return G, D

def test(G):
	Z=np.random.normal(loc=0.0, scale=1.0, size=(100,100)).astype(floatX)
	G_Z=get_output(G,Z)
	return G_Z


opts = get_args()
print opts

# x_train=loadData()
# G,D=train(x_train)
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




