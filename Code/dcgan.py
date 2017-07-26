#DCGAN Using Lasagne (for CelebA)
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

from functions import get_args, load_CelebA
from nets import get_gen, get_dis

import os

floatX=theano.config.floatX


def build_net(nz=100):
	# nz = size of latent code
	gen = get_gen(nz=nz)
	dis = get_dis(nz=nz)

	return gen, dis


def prep_train(lr=0.0002, nz=100):
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

def train(nz=100, lr=0.001, batchSize=64, epoch=10):


	xTrain = load_CelebA()
	train_fns, test_fns, G, D = prep_train(nz=nz, lr=lr)

	sn,sc,sx,sy=np.shape(trainData)
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
	plt.savefig(os.path.join(opts.outDir,'/cost_regular.png'))

	return G, D

def test(G):
	Z=np.random.normal(loc=0.0, scale=1.0, size=(100,100)).astype(floatX)
	G_Z=get_output(G,Z)
	return G_Z



if __name__ == '__main__':
	opts = get_args()
	G,D=train(nz=opts.nz, lr=opts.lr, batchSize=opts.batchSize, epoch=opts.maxEpochs)



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




