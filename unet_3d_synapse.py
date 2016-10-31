import keras
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Convolution3D, MaxPooling3D, UpSampling3D, merge, ZeroPadding2D, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import SGD
from keras.regularizers import l2
from prepare_data_synapse import *
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import random
import scipy.misc
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from theano.ifelse import ifelse

rng = np.random.RandomState(7)

learning_rate = 0.001
momentum = 0.99
doTrain = int(sys.argv[1])
doFineTune = False

patchSize = 508
patchSize_out = 324
patchZ = 12
patchZ_out = 4
cropSize = (patchSize-patchSize_out)/2
csZ = (patchZ-patchZ_out)/2

weight_decay = 0.
weight_class_1 = 1.

patience = 100
patience_reset = 100
doBatchNormAll = True
doDropout = False

purpose = 'train'
initialization = 'glorot_uniform'
filename = '3d_unet_synapse'
numKernel = 32

srng = RandomStreams(1234)

# need to define a custom loss, because all pre-implementations
# seem to assume that scores over patch add up to one which
# they clearly don't and shouldn't
def unet_crossentropy_loss(y_true, y_pred):
    epsilon = 1.0e-4
    y_pred_clipped = T.clip(y_pred, epsilon, 1.0-epsilon)
    loss_vector = -T.mean(weight_class_1*y_true * T.log(y_pred_clipped) + (1-y_true) * T.log(1-y_pred_clipped), axis=1)
    average_loss = T.mean(loss_vector)
    return average_loss

def unet_crossentropy_loss_sampled(y_true, y_pred):
    # weighted version of pixel-wise crossrntropy loss function
    alpha = 0.6
    epsilon = 1.0e-5
    y_pred_clipped = T.flatten(T.clip(y_pred, epsilon, 1.0-epsilon))
    y_true = T.flatten(y_true)
    # this seems to work
    # it is super ugly though and I am sure there is a better way to do it
    # but I am struggling with theano to cooperate
    # filter the right indices
    indPos = T.nonzero(y_true)[0] # no idea why this is a tuple
    indNeg = T.nonzero(1-y_true)[0]
    # shuffle
    n = indPos.shape[0]
    indPos = indPos[srng.permutation(n=n)]
    n = indNeg.shape[0]
    indNeg = indNeg[srng.permutation(n=n)]
   
    # take equal number of samples depending on which class has less
    n_samples = T.cast(T.min([T.sum(y_true), T.sum(1-y_true)]), dtype='int64')
    # indPos = indPos[:n_samples]
    # indNeg = indNeg[:n_samples]

    total = np.float64(patchSize_out*patchSize_out*patchZ_out)
    loss_vector = ifelse(T.gt(n_samples, 0),
			 # if this patch has positive samples, then calulate the first formula
			 (- alpha*T.sum(T.log(y_pred_clipped[indPos])) - (1-alpha)*T.sum(T.log(1-y_pred_clipped[indNeg])))/total, 
			 - (1-alpha)*T.sum(T.log(1-y_pred_clipped[indNeg]))/total )

    average_loss = T.mean(loss_vector)/(1-alpha)
    return average_loss


def unet_block_down(input, nb_filter, doPooling=True, doDropout=False, doBatchNorm=False, downsampleZ=False, thickness1=1, thickness2=1):
    # first convolutional block consisting of 2 conv layers plus activation, then maxpool.
    # All are valid area, not same
    act1 = Convolution3D(nb_filter=nb_filter, kernel_dim1=thickness1, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),
                         init=initialization, activation='relu',  border_mode="valid", W_regularizer=l2(weight_decay))(input)
    if doBatchNorm:
        act1 = BatchNormalization(mode=0, axis=1)(act1)

    act2 = Convolution3D(nb_filter=nb_filter, kernel_dim1=thickness2, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),
                         init=initialization, activation='relu',  border_mode="valid", W_regularizer=l2(weight_decay))(act1)
    if doBatchNorm:
        act2 = BatchNormalization(mode=0, axis=1)(act2)

    if doDropout:
        act2 = Dropout(0.5)(act2)

    if doPooling:
        # now downsamplig with maxpool
        if downsampleZ:
            pool1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode="valid")(act2)

        else:
            pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode="valid")(act2)

    else:
        pool1 = act2

    return (act2, pool1)


# need to define lambda layer to implement cropping
def crop_layer(x, cs, csZ):
    cropSize = cs
    if csZ == 0:
        return x[:,:,:,cropSize:-cropSize, cropSize:-cropSize]
    else:	
        return x[:,:,csZ:-csZ,cropSize:-cropSize, cropSize:-cropSize]


def unet_block_up(input, nb_filter, down_block_out, doBatchNorm=False, upsampleZ=False, thickness1=1, thickness2=1):
    print "This is unet_block_up"
    print "input ", input._keras_shape
    # upsampling
    if upsampleZ:
        up_sampled = UpSampling3D(size=(2,2,2))(input)
    else:
        up_sampled = UpSampling3D(size=(1,2,2))(input)
    print "upsampled ", up_sampled._keras_shape
    # up-convolution
    conv_up = Convolution3D(nb_filter=nb_filter, kernel_dim1=2, kernel_dim2=2, kernel_dim3=2, subsample=(1,1,1),
                            init=initialization, activation='relu', border_mode="same", W_regularizer=l2(weight_decay))(up_sampled)
    print "up-convolution ", conv_up._keras_shape
    # concatenation with cropped high res output
    # this is too large and needs to be cropped
    print "to be merged with ", down_block_out._keras_shape

    cropSize = int((down_block_out._keras_shape[3] - conv_up._keras_shape[3])/2)
    csZ      = int((down_block_out._keras_shape[2] - conv_up._keras_shape[2])/2)
    # input is a tensor of size (batchsize, channels, thickness, width, height)
    down_block_out_cropped = Lambda(crop_layer, output_shape=conv_up._keras_shape[1:], arguments={"cs":cropSize,"csZ":csZ})(down_block_out)
    print "cropped layer size: ", down_block_out_cropped._keras_shape
    merged = merge([conv_up, down_block_out_cropped], mode='concat', concat_axis=1)

    print "merged ", merged._keras_shape
    act1 = Convolution3D(nb_filter=nb_filter, kernel_dim1=thickness1, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),
                         init=initialization, activation='relu',  border_mode="valid", W_regularizer=l2(weight_decay))(merged)

    if doBatchNorm:
        act1 = BatchNormalization(mode=0, axis=1)(act1)

    print "conv1 ", act1._keras_shape
    act2 = Convolution3D(nb_filter=nb_filter, kernel_dim1=thickness2, kernel_dim2=3, kernel_dim3=3, subsample=(1,1,1),
                         init=initialization, activation='relu',  border_mode="valid", W_regularizer=l2(weight_decay))(act1)
    if doBatchNorm:
        act2 = BatchNormalization(mode=0, axis=1)(act2)

    print "conv2 ", act2._keras_shape

    return act2


if doTrain: # remember doTrain = int(sys.argv[1])
    # input data should be large patches as prediction is also over large patches
    print
    print "==== building network ===="
    print

    print "== BLOCK 1 =="
    input = Input(shape=(1, patchZ, patchSize, patchSize))
    print "input  ", input._keras_shape
    block1_act, block1_pool = unet_block_down(input=input, nb_filter=numKernel, doBatchNorm=doBatchNormAll, thickness1=2, thickness2=2, downsampleZ=False)
    print "block1 ", block1_pool._keras_shape

    print "== BLOCK 2 =="
    block2_act, block2_pool = unet_block_down(input=block1_pool, nb_filter=numKernel*2, doBatchNorm=doBatchNormAll, thickness1=1, thickness2=1, downsampleZ=False)
    print "block2 ", block2_pool._keras_shape

    print "== BLOCK 3 =="
    block3_act, block3_pool = unet_block_down(input=block2_pool, nb_filter=numKernel*4, doBatchNorm=doBatchNormAll, thickness1=2, thickness2=2, downsampleZ=True)
    print "block3 ", block3_pool._keras_shape

    print "== BLOCK 4 =="
    block4_act, block4_pool = unet_block_down(input=block3_pool, nb_filter=numKernel*8, doBatchNorm=doBatchNormAll, thickness1=1, thickness2=1, downsampleZ=True)
    print "block4 ", block4_pool._keras_shape

    print "== BLOCK 5 =="
    print "#no pooling for the bottom layer"
    block5_act, block5_pool = unet_block_down(input=block4_pool, nb_filter=numKernel*16, doPooling=False, doBatchNorm=doBatchNormAll, thickness1=1, thickness2=1)
    print "block5 ", block5_pool._keras_shape

    print
    print "=============="
    print

    print "== BLOCK 4 UP =="
    block4_up = unet_block_up(input=block5_act, nb_filter=numKernel*8, down_block_out=block4_act, doBatchNorm=doBatchNormAll, upsampleZ=True, thickness1=1, thickness2=1)
    print
    print "== BLOCK 3 UP =="
    block3_up = unet_block_up(input=block4_up,  nb_filter=numKernel*4, down_block_out=block3_act, doBatchNorm=doBatchNormAll, upsampleZ=True, thickness1=2, thickness2=2)
    print
    print "== BLOCK 2 UP =="
    block2_up = unet_block_up(input=block3_up,  nb_filter=numKernel*2, down_block_out=block2_act, doBatchNorm=doBatchNormAll, upsampleZ=False,thickness1=1, thickness2=1)
    print
    print "== BLOCK 1 UP =="
    block1_up = unet_block_up(input=block2_up,  nb_filter=numKernel*1, down_block_out=block1_act, doBatchNorm=doBatchNormAll, upsampleZ=False,thickness1=2, thickness2=2)
    print
    print "== 1x1 convolution =="
    output = Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1, subsample=(1,1,1),
                           init=initialization, activation='sigmoid', border_mode="valid")(block1_up)

    print "output ", output._keras_shape
    output_flat = Flatten()(output)
    print "output flat ", output_flat._keras_shape
    print

    if doFineTune:
        model = model_from_json(open('3d_unet_synapse_best.json').read())
        model.load_weights('3d_unet_synapse_best_weights.h5')
	print 'use previous parameters'
	print

    else:
    	model = Model(input=input, output=output_flat)

    sgd = SGD(lr=learning_rate, decay=0, momentum=momentum, nesterov=False)
    model.compile(loss=unet_crossentropy_loss_sampled, optimizer=sgd)
    #finish the construcution of 3D unet model for spine detection

    best_performance_so_far = 0.

    patience_counter = 0
    for epoch in xrange(10000000):
	print
        print "Waiting for data."

	data = generate_sample_synapse(purpose='train', nsamples_patch=4, nsamples_block=5, doAugmentation=True,
				     patchSize=patchSize, patchSize_out=patchSize_out, patchZ=patchZ, patchZ_out=patchZ_out)

        data_x = data[0].astype(np.float32)
        data_x = np.reshape(data_x, [-1, 1, patchZ, patchSize, patchSize])
        data_y = data[1].astype(np.float32)
	del data

	print 
	print "Data_x shape: ", data_x.shape
	print "Data_y shape: ", data_y.shape
        print "current learning rate: ", model.optimizer.lr.get_value()

	for k in range(data_x.shape[0]):
	    X = data_x[k:k+1]
	    Y = data_y[k:k+1]
	    print "Unique:--", np.unique(Y)
	    if np.unique(Y).shape[0] == 2:
	        model.fit(X, Y, batch_size=1, nb_epoch=1)

        json_string = model.to_json()
        open(filename+'.json', 'w').write(json_string)
        model.save_weights(filename+'_weights.h5', overwrite=True)
	del data_x, data_y

        Probas_, Labels_ = prediction_full_patch_synapse(patchSize=patchSize, patchSize_out=patchSize_out, patchZ=patchZ, patchZ_out=patchZ_out, returnValue=True)
        numImage = Probas_.shape[0]

        temp_best_metric = 0.
        for k in range(2, 9):
            thresh = 0.1*k
            mean_val_metric = 0.
            Temp_ = Probas_.copy()
            for imgIndex in range(numImage):
                for index in range(Probas_.shape[1]):
                    if Probas_[imgIndex][index] > thresh:
                        Temp_[imgIndex][index] = 1.0
                    else:
                        Temp_[imgIndex][index] = 0.0

            for imgIndex in range(numImage):
                #calculate F1 score for evaluation on validation images
                average_f_score = f1_score(Labels_[imgIndex], Temp_[imgIndex], average='micro')
                mean_val_metric += average_f_score

            mean_val_metric = mean_val_metric/float(numImage)
            if mean_val_metric > temp_best_metric:
                temp_best_metric = mean_val_metric

        mean_val_metric = temp_best_metric
	fp = open("F1_score.txt","a")
        fp.write(str(mean_val_metric)+"\n")
        fp.close()

        print mean_val_metric, " > ",  best_performance_so_far, "?"
        print mean_val_metric - best_performance_so_far
        if  mean_val_metric > best_performance_so_far:
            best_performance_so_far = mean_val_metric
            print "NEW BEST MODEL"
            json_string = model.to_json()
            open(filename+'_best.json', 'w').write(json_string)
            model.save_weights(filename+'_best_weights.h5', overwrite=True)
            patience_counter =0
        else:
            patience_counter+=1

        # no progress anymore, need to decrease learning rate
        if patience_counter == patience:
            print "DECREASING LEARNING RATE"
            print "before: ", learning_rate
            learning_rate *= 0.1
            print "now: ", learning_rate
            model.optimizer.lr.set_value(learning_rate)
            patience = patience_reset
            patience_counter = 0

            # reload best state seen so far
            model = model_from_json(open(filename+'_best.json').read())
            model.load_weights(filename+'_best_weights.h5')
            model.compile(loss=unet_crossentropy_loss_sampled, optimizer=sgd)

        # stop if not learning anymore
        if learning_rate < 1e-7:
            break
