### Script to train a conditional adversarial autoencoder

#Imports
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, flatten,\
reshape, batch_norm, Upscale2DLayer
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import sigmoid, softmax
from lasagne.layers import get_output, get_all_params, get_output_shape, get_all_layers
from lasagne.objectives import binary_crossentropy as bce
from lasagne.objectives import categorical_crossentropy as cce
from lasagne.objectives import squared_error
from lasagne.updates import adam, sgd, total_norm_constraint, momentum
from lasagne.regularization import l2, l1
from lasagne.regularization import regularize_layer_params
from lasagne.objectives import categorical_accuracy as acc

import numpy as np
import theano
from theano import tensor as T
import time
from matplotlib import pyplot as plt 

from skimage.io import imsave

import pickle

import os, sys

floatX=theano.config.floatX

from functions import get_args, load_MNIST
from nets import get_enc_MNIST, get_Zenc_MNIST, get_Yenc_MNIST, get_dec_MNIST, get_disZ_MNIST

import argparse


#class with training and model options
class opts(object):
	def __init__(self):
		self.nz=15
		self.imSize=28*28
		self.gradClipping=10
		self.beta=0.8

		args=get_args()
		self.outDir = args.outDir
		self.batchSize = args.batchSize
		self.lr = args.lr
		self.maxEpochs= args.maxEpochs


def build_nets(opts):
	# image --> encoding
	enc = get_enc_MNIST()

	# encoding --> latent rep
	Zenc = get_Zenc_MNIST(nz=opts.nz)
	
	#encoding --> label vector
	Yenc = get_Yenc_MNIST()

	#[latent , label] --> sample
	dec = get_dec_MNIST(nz=opts.nz)

	# z --> real or fake
	disZ = get_disZ_MNIST(nz=opts.nz)


	# y --> real or fake
	disY = get_disY_MNIST()

	return enc, Zenc, Yenc, dec, disZ, disY


def prep_train(opts):

	#vars
	x = T.matrix('x') #input images
	y = T.matrix('y') #labels
	z = T.matrix('z')
	lrREC = T.scalar('lrREC') #recosntruction lr
	lrADV = T.scalar('lrADV') #adversarial lr
	lrSUP = T.scalar('lrSUP') #supervised lr

	#get nets
	enc, Zenc, Yenc, dec, disZ, disY = build_nets(opts)

	#get outputs for training
	encoding = get_output(enc,x)
	
	zFake = get_output(Zenc, encoding)
	yFake = get_output(Yenc, encoding)

	rec = get_output(dec, T.concatenate([zFake, yFake], axis=1))

	pZreal = get_output(disZ, z)
	pZfake = get_output(disZ, zFake)

	pYreal = get_output(disY, y)
	pYfake = get_output(disY, yFake)

	
	#get output for testing   *leave out sampling for now just reconstruction and clasification

	#loss functions
	#reconstruction loss
	J_rec = T.mean(squared_error(rec, x))

	#adversarial loss Y
	J_adv_disZ = T.mean(bce(pZreal, T.ones(pZreal.shape))) + T.mean(bce(pZfake, T.zeros(pZfake.shape)))
	J_adv_encZ = T.mean(bce(pZfake, T.ones(pZfake.shape)))

	#adversarial loss Z
	J_adv_disY = T.mean(bce(pYreal, T.ones(pYreal.shape))) + T.mean(bce(pYfake, T.zeros(pYfake.shape)))
	J_adv_encY = T.mean(bce(pYfake, T.ones(pYfake.shape)))

	J_enc = J_adv_encY + J_adv_encZ
	J_dis = J_adv_disY + J_adv_disZ

	#supervised loss
	MIN = 1e-6
	MAX = 1-MIN
	J_class = T.mean(cce(T.clip(yFake,MIN, MAX), y))

	#get params
	params_enc = get_all_params(enc, trainable=True) \
		+ get_all_params(Zenc, trainable=True) \
		+ get_all_params(Yenc, trainable=True)
	params_dec = get_all_params(dec, trainable=True)
	params_dis = get_all_params(disZ, trainable=True)\
		+ get_all_params(disY, trainable=True)

	#get grads
	grad_enc = total_norm_constraint(T.grad(J_enc, params_enc),opts.gradClipping)
	grad_rec = total_norm_constraint(T.grad(J_rec, params_dec+params_enc),opts.gradClipping)
	grad_dis = total_norm_constraint(T.grad(J_dis, params_dis),opts.gradClipping)

	#get updates
	update_enc = momentum(grad_enc, params_enc, learning_rate=lrADV, momentum=0.1)
	update_rec = momentum(grad_rec, params_dec+params_enc, learning_rate=lrREC, momentum=0.9)
	update_dis = momentum(grad_dis, params_dis, learning_rate=lrADV, momentum=0.1)
	update_class = momentum(J_class, get_all_params(enc, trainable=True) \
		+ get_all_params(Yenc, trainable=True), learning_rate=lrSUP, momentum=0.9)
	#update_class = adam(J_class, get_all_params(Yenc, trainable=True), learning_rate=opts.lr, beta1=opts.beta)  #N.B. could also update enc

	#theano train functions
	train_fns={}
	train_fns['enc'] = theano.function(inputs=[x, lrADV], outputs=[J_enc, J_adv_encY, J_adv_encZ], updates=update_enc)
	train_fns['rec'] = theano.function(inputs=[x, lrREC], outputs=J_rec, updates=update_rec)
	train_fns['dis'] = theano.function(inputs=[x,y,z, lrADV], outputs=[J_dis, J_adv_disZ,J_adv_disY], updates=update_dis)
	train_fns['class'] = theano.function(inputs=[x,y, lrSUP], outputs=J_class, updates=update_class)

	#theano test functions
	test_fns={}
	test_fns['rec'] = theano.function(inputs=[x], outputs=rec)
	test_fns['encZ'] = theano.function(inputs=[x], outputs=zFake)
	samples = get_output(dec, T.concatenate([z,y], axis=1))
	test_fns['gen'] = theano.function(inputs=[y,z], outputs=samples)
	test_fns['rec_error'] = theano.function(inputs=[x], outputs=squared_error(x,rec))
	test_fns['predictY'] = theano.function(inputs=[x], outputs=yFake)
	test_fns['accY'] = theano.function(inputs=[x,y], outputs=acc(yFake, y))


	return train_fns, test_fns

def Pz(noSamples, nz=10):
	z = np.random.normal(size=(noSamples, nz))
	return z.astype(floatX)


def train(opts):

	train_fns, test_fns = prep_train(opts)
	x_train, y_train, x_test, y_test, x_val, y_val =load_MNIST(opts)

	print 'Training data loaded: no test healthy images=',np.sum(y_test), '. no unhealthy images=', y_test.shape[0]-np.sum(y_test)

	train_costs={'enc':[], 'rec':[], 'disZ':[], 'disY':[], 'accY':[], 'dis':[]}
	test_costs={'rec':[], 'accY':[]}

	noBatches=x_train.shape[0]//opts.batchSize
	for e in range(opts.maxEpochs):

		if e%50:
			lrREC, lrADV, lrSUP = 0.001, 0.01, 0.01
		elif e%1000:
			lrREC, lrADV, lrSUP = 0.0001, 0.001, 0.001
		else:
			lrREC, lrADV, lrSUP = 0.00001, 0.0001, 0.0001 

		print 'epoch \t batch \t\t enc \t\t dec \t\t dis \t\t test_rec \t\t test_acc \t\t train_acc'
		for b in range(noBatches):

			X = x_train[b*opts.batchSize:(b+1)*opts.batchSize]
			Y = np.eye(10)[y_train[b*opts.batchSize:(b+1)*opts.batchSize]].astype(floatX)  #one hot encodings
			Z = Pz(opts.batchSize, nz=opts.nz)

			J_rec=train_fns['rec'](X, lrREC)
			J_dis, J_disZ, J_disY=train_fns['dis'](X,Y,Z, lrADV)
			J_enc, J_encY, J_encZ=train_fns['enc'](X, lrADV)
			J_class = train_fns['class'](X,Y, lrSUP)
			train_accY = 100*test_fns['accY'](X,Y).mean()


			train_costs['enc'].append(J_enc)
			train_costs['rec'].append(J_rec)
			train_costs['dis'].append(J_dis)
			train_costs['disZ'].append(J_disZ)
			train_costs['disY'].append(J_disY)
			train_costs['accY'].append(train_accY)

			if b%100==0:
				print e,'/', opts.maxEpochs,'\t', b,'/',noBatches,'\t', J_enc, '\t', J_rec, '\t', J_dis, '\t\t _ \t\t _'
				print 'J_encY:', J_encY, 'J_encZ:', J_encZ, 'J_disY:', J_disY, 'J_disZ', J_disZ

			if b%100==0:
				X = x_test[:opts.batchSize]
				Y = np.eye(10)[y_test[:opts.batchSize]].astype(floatX)
				J_rec_test = test_fns['rec_error'](X).mean()
				test_accY = 100*test_fns['accY'](X,Y).mean()

				test_costs['rec'].append(J_rec)
				test_costs['accY'].append(test_accY)



				print '_ \t\t\t _ \t\t _ \t\t _ \t\t', J_rec_test, '\t\t', test_accY, '\t\t', train_accY

	return train_costs, test_costs, train_fns, test_fns

#Function to apply learned models to data in batches
def apply(model, data, batchSize=64):
	output=[]
	for b in range(np.shape(data)[0]//batchSize):
		out=model(data[b*batchSize:(b+1)*batchSize])
		output.append(out)
	return np.concatenate(output)


def save(opts, train_costs, test_costs, train_fns, test_fns):
	outDir = opts.outDir

	i=1
	while os.path.isdir(os.path.join(outDir,'Ex_'+str(i))):
		i+=1
	newDir=os.path.join(outDir,'Ex_'+str(i))
	os.mkdir(newDir)

	#train and test axies
	noTrainIter=len(train_costs['rec'])
	noTestIter=len(test_costs['rec'])
	trainAxis = range(noTrainIter)
	step=float(noTrainIter)/noTestIter
	testAxis = np.arange(0,noTrainIter, step)

	#Save reconstruction plot
	fig1 = plt.figure()
	plt.plot(trainAxis,train_costs['rec'], label='train')
	plt.plot(testAxis, test_costs['rec'], label='test')
	plt.legend()
	plt.title('Reconstruction cost')
	plt.xlabel('batch no.')
	plt.ylabel('mean sqaured error')
	plt.savefig(os.path.join(newDir,'recPlot.png'))

	#Save acc plot
	fig2 = plt.figure()
	plt.plot(trainAxis, train_costs['accY'], label='train')
	plt.plot(testAxis, test_costs['accY'], label='test')
	plt.legend()
	plt.title('Classiciation accuracy')
	plt.xlabel('batch no.')
	plt.ylabel('classification accuracy %')
	plt.savefig(os.path.join(newDir,'accPlot.png'))

	#Save enc, dec, dis training loss
	fig3 = plt.figure()
	plt.plot(trainAxis, train_costs['enc'], label='enc.')
	plt.plot(trainAxis, train_costs['dis'], label='dis.')
	plt.legend()
	plt.title('Training losses')
	plt.xlabel('batch no.')
	plt.ylabel('loss')
	plt.savefig(os.path.join(newDir, 'trainLoss.png'))


	#load data again:
	x_train, y_train, x_test, y_test, x_val, y_val = load_data(opts)
	
	#Save example reconstructions
	rec = apply(test_fns['rec'], x_test)
	print np.shape(rec), np.shape(x_test)
	try:
		test_rec = np.mean((rec - x_test[:rec.shape[0]])**2)
		print 'Mean Test Reconstruction Error:', test_rec  #may no rec all
	except:
		print 'error calc mse'

	#Save generated samples (by category)
	fig4=plt.figure()
	montageRow1 = np.hstack(x_test[:5].reshape(-1,28,28))
	montageRow2 = np.hstack(rec[:5].reshape(-1,28,28))
	montage = np.vstack((montageRow1, montageRow2))
	plt.imshow(montage, cmap='gray')
	plt.savefig(os.path.join(newDir,'rec.png'))

	fig5 = plt.figure()
	montage=[]
	for i in range(10):
		Y = np.zeros((5,10))
		Y[:,i]=np.ones((5))
		Y=Y.astype(floatX)
		Z = Pz(5,opts.nz)
		samples = np.hstack((test_fns['gen'](Y,Z)).reshape(-1,28,28))  #small enough not to need apply fn
		montage.append(samples)
	plt.imshow(np.vstack(montage), cmap='gray')
	plt.savefig(os.path.join(newDir,'samples.png'))

	fig6 = plt.figure()
	encZ = apply(test_fns['encZ'], x_test)
	noSamples=10**4
	H, edges = np.histogram(Pz(noSamples),100)
	plt.plot(edges[:-1], H.astype(float)/noSamples, label='Prior')
	for i in range(encZ.shape[1]):
		H,_ = np.histogram(encZ[:,i], edges)
		plt.plot(edges[:-1], H.astype(float)/encZ.shape[0], label='enc channel'+str(i))
	plt.xlabel('Encoding value')
	plt.ylabel('Probability')
	plt.title('Distribution of encoded samples')
	plt.legend()
	plt.savefig(os.path.join(newDir,'enc&priorPlot.png'))

	fig7 = plt.figure()
	for i in range(encZ.shape[1]):
		H,edges = np.histogram(encZ[:,i], 100)
		plt.plot(edges[:-1], H, label='enc channel'+str(i))
	plt.xlabel('Encoding value')
	plt.ylabel('Probability')
	plt.title('Distribution of encoded samples')
	plt.legend()
	plt.savefig(os.path.join(newDir,'encPlot.png'))

	#Save accuracy and recosntruction error for all test samples in this file
	f = open(os.path.join(newDir,'outputs.txt'), 'w')
	f.write('Test reconstruction error:'+str(test_rec))
	test_acc = test_fns['accY'](x_test, np.eye(10)[y_test].astype(floatX))
	f.write('Test classification accuray:'+str(np.mean(test_acc)))
	f.close()




	#Save opts and functions (opts, train_costs, test_costs, train_fns, test_fns)
	sys.setrecursionlimit(10000)
	pickle.dump(opts, open(os.path.join(newDir, 'opts.pkl'),'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_costs, open(os.path.join(newDir, 'train_costs.pkl'),'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_costs, open(os.path.join(newDir, 'test_costs.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_fns, open(os.path.join(newDir, 'train_fns.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_fns, open(os.path.join(newDir, 'test_fns.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == '__main__':
	myOpts=opts()
	train_costs, test_costs, train_fns, test_fns = train(myOpts)
	save(myOpts, train_costs, test_costs, train_fns, test_fns)






