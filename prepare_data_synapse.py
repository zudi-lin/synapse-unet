import time
import glob
import mahotas
import numpy as np
import scipy.misc
import random
from keras.models import Model, Sequential, model_from_json
import math


def normalizeImage(img, saturation_level=0.05):
    sortedValues = np.sort( img.ravel())
    minVal = np.float32(sortedValues[np.int(len(sortedValues) * (saturation_level / 2))])
    maxVal = np.float32(sortedValues[np.int(len(sortedValues) * (1 - saturation_level / 2))])
    normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))
    normImg[normImg<0] = 0
    normImg[normImg>255] = 255
    return (np.float32(normImg) / 255.0)


def shuffle_together(data_set):
    xlist = range(data_set[0].shape[0]) #total number of samples
    xlist = random.sample(xlist,len(xlist))
    new_data_set = []
    for module in range(len(data_set)): # number of data modules
        newdata = data_set[module].copy()
        for dataIndex in range(len(xlist)):
            newdata[dataIndex] = data_set[module][xlist[dataIndex]]
        new_data_set.append(newdata)
    return new_data_set


def mirror_image_layer(img, cropSize=92):
    mirror_image = np.zeros((img.shape[0]+2*cropSize, img.shape[0]+2*cropSize))
    length = img.shape[0]
    mirror_image[cropSize:cropSize+length,cropSize:cropSize+length]=img
    mirror_image[0:cropSize,0:cropSize]=np.rot90(img[0:cropSize,0:cropSize],2)
    mirror_image[-cropSize:,0:cropSize]=np.rot90(img[-cropSize:,0:cropSize],2)
    mirror_image[0:cropSize,-cropSize:]=np.rot90(img[0:cropSize,-cropSize:],2)
    mirror_image[-cropSize:,-cropSize:]=np.rot90(img[-cropSize:,-cropSize:],2)

    mirror_image[0:cropSize,cropSize:cropSize+length]=np.flipud(img[0:cropSize,0:length])
    mirror_image[cropSize:cropSize+length,0:cropSize]=np.fliplr(img[0:length,0:cropSize])
    mirror_image[cropSize:cropSize+length,-cropSize:]=np.fliplr(img[0:length,-cropSize:])
    mirror_image[-cropSize:,cropSize:cropSize+length]=np.flipud(img[-cropSize:,0:length])

    return mirror_image


def crop_image_layer(x, cs):
    cropSize = cs
    return x[cropSize:-cropSize, cropSize:-cropSize]


def generate_sample_spine(purpose='train', nsamples_patch=5, nsamples_block=10, patchSize=572, patchSize_out=388, patchZ=23, patchZ_out=1, block_name='train', doAugmentation=False):

    start_time = time.time()
    pathPrefix = '/n/home09/zlin2016/project/spine_detection/spine_model_1.1/'
    img_search_string_grayImages     = pathPrefix + 'images/' + purpose + '/*.png'
    img_search_string_membraneImages = pathPrefix + 'spinemasks/'+ purpose + '/*.png'

    img_files_gray     = sorted( glob.glob( img_search_string_grayImages ) )
    img_files_membrane = sorted( glob.glob( img_search_string_membraneImages ) )

    cropSize = (patchSize - patchSize_out)/2
    csZ = (patchZ - patchZ_out)/2
    print 'crop size: ', cropSize
    print 'crop thickness: ', csZ

    img = mahotas.imread(img_files_gray[0])#read the first image to get imformation about the shape
    grayImages    = np.zeros((np.shape(img_files_gray)[0], img.shape[0], img.shape[1]))
    membraneImages= np.zeros((np.shape(img_files_gray)[0], img.shape[0], img.shape[1]))
    
    read_order = range(np.shape(img_files_gray)[0])
    for img_index in read_order:
        img = mahotas.imread(img_files_gray[img_index])
        img = normalizeImage(img)
	img = img-0.5
	grayImages[img_index,:,:] = img
        membrane_img = mahotas.imread(img_files_membrane[img_index])/255.
        membraneImages[img_index,:,:] = membrane_img

    if doAugmentation:
    	nsamples = 6*nsamples_block*nsamples_patch
    else:
        nsamples = 1*nsamples_block*nsamples_patch

    grayImg_set = np.zeros((nsamples, patchZ, patchSize, patchSize))
    membrane_set= np.zeros((nsamples, patchZ_out, patchSize_out, patchSize_out))

    pickIndex=random.sample(range(0, np.shape(img_files_gray)[0]-patchZ+1), nsamples_block)
    num_total = 0
    for i in pickIndex:
        for j in range(nsamples_patch):
	    x_index = random.randint(0,img.shape[0]-patchSize)
	    y_index = random.randint(0,img.shape[0]-patchSize)
	    grayImg_set[num_total,:,:,:] = grayImages[i:i+patchZ, x_index:x_index+patchSize, y_index:y_index+patchSize]
	    membrane_set[num_total,:,:,:]= membraneImages[i+csZ:i+csZ+patchZ_out, x_index+cropSize:x_index+cropSize+patchSize_out, y_index+cropSize:y_index+cropSize+patchSize_out]

	    num_total += 1
            
 	    if doAugmentation:
	        temp_gray = np.zeros((patchSize, patchSize))
	        temp_label= np.zeros((patchSize_out, patchSize_out))
                # augmentation through rotation & flip
	        for k in range(1,4):
		    for n in range(patchZ_out):
		        temp_label = membraneImages[i+csZ+n, x_index+cropSize:x_index+cropSize+patchSize_out, y_index+cropSize:y_index+cropSize+patchSize_out]
	                membrane_set[num_total,n,:,:] = np.rot90(temp_label,k)
		    for m in range(patchZ):
	 	        temp_gray = grayImages[i+m, x_index:x_index+patchSize, y_index:y_index+patchSize]
		        grayImg_set[num_total,m,:,:] = np.rot90(temp_gray,k)

		    num_total += 1

		for n in range(patchZ_out):
	            temp_label = membraneImages[i+csZ+n, x_index+cropSize:x_index+cropSize+patchSize_out, y_index+cropSize:y_index+cropSize+patchSize_out]
	            membrane_set[num_total,n,:,:] = np.fliplr(temp_label)
	        for m in range(patchZ):
		    temp_gray = grayImages[i+m, x_index:x_index+patchSize, y_index:y_index+patchSize]
	            grayImg_set[num_total,m,:,:] = np.fliplr(temp_gray)

	        num_total += 1

		for n in range(patchZ_out):
	            temp_label = membraneImages[i+csZ+n, x_index+cropSize:x_index+cropSize+patchSize_out, y_index+cropSize:y_index+cropSize+patchSize_out]
	            membrane_set[num_total,n,:,:] = np.flipud(temp_label)
	        for m in range(patchZ):
		    temp_gray = grayImages[i+m, x_index:x_index+patchSize, y_index:y_index+patchSize]
	            grayImg_set[num_total,m,:,:] = np.flipud(temp_gray)

	        num_total += 1


    print 'Total number of training samples: ', num_total

    newMembrane = np.zeros((num_total, patchZ_out*patchSize_out*patchSize_out))
    for i in range(num_total):
	newMembrane[i] = membrane_set[i].flatten()

    data_set = (grayImg_set, newMembrane)
    data_set = shuffle_together(data_set)

    end_time = time.time()
    total_time = (end_time - start_time)

    print 'Running time: ', total_time / 60.
    print 'finished sampling data'

    return data_set


def prediction_full_patch_spine(patchSize=572, patchSize_out=388, patchZ=23, patchZ_out=1, writeImage=True, returnValue=True):
    
    start_time = time.time()
    pathPrefix = '/n/home09/zlin2016/project/spine_detection/spine_model_1.1/'
    img_search_string_grayImages     = pathPrefix + 'images/validate/*.png'
    img_search_string_membraneImages = pathPrefix + 'spinemasks/validate/*.png'
    img_files_gray = sorted(glob.glob( img_search_string_grayImages ))
    img_files_membrane = sorted( glob.glob( img_search_string_membraneImages ))

    #load model
    print 'Read the model for evaluation'
    model = model_from_json(open('3d_unet_spine.json').read())
    model.load_weights('3d_unet_spine_weights.h5')

    cropSize = (patchSize - patchSize_out)/2
    csZ = (patchZ - patchZ_out)/2

    img = mahotas.imread(img_files_gray[0])#read the first image to get imformation about the shape
    grayImages = np.zeros((np.shape(img_files_gray)[0], img.shape[0], img.shape[1]))
    labelImages= np.zeros((np.shape(img_files_gray)[0], img.shape[0], img.shape[1]),dtype=np.int8)
    probImages = np.zeros((np.shape(img_files_gray)[0]-2*csZ, img.shape[0]-2*cropSize, img.shape[1]-2*cropSize))

    print 'Total number of full size test images:', np.shape(img_files_gray)[0]
    read_order = range(np.shape(img_files_gray)[0])
    for img_index in read_order:
        img = mahotas.imread(img_files_gray[img_index])
        img = normalizeImage(img)
	img = img-0.5
	grayImages[img_index,:,:] = img
        img_label = mahotas.imread(img_files_membrane[img_index])/255
        labelImages[img_index,:,:]= np.int8(img_label)
        
    numSample_axis  = int((img.shape[0]-2*cropSize)/patchSize_out)+1
    numSample_patch = numSample_axis**2
    numZ = float(len(img_files_gray)-2*csZ)/float(patchZ_out)
    numZ = int(math.ceil(numZ))
    nsamples = numSample_patch*numZ
    print 'Number of inputs for this block:', nsamples

    grayImg_set = np.zeros((nsamples, patchZ, patchSize, patchSize))
    membrane_set= np.zeros((nsamples, patchZ_out, patchSize_out, patchSize_out))

    print 'Total number of probability maps:', len(img_files_gray)-2*csZ
    numProb = len(img_files_gray)-2*csZ   

    num_total = 0
    for zIndex in range(numZ):
        if zIndex == numZ-1:
	    zStart = numProb-patchZ_out
	else:
	    zStart = patchZ_out*zIndex

	for xIndex in range(numSample_axis-1):
	    xStart = patchSize_out*xIndex
	    for yIndex in range(numSample_axis-1):
		yStart = patchSize_out*yIndex
		grayImg_set[num_total] = grayImages[zStart:zStart+patchZ, xStart:xStart+patchSize, yStart:yStart+patchSize]
		num_total += 1

	xStart = img.shape[0]-patchSize
	for yIndex in range(numSample_axis-1):
	    yStart = patchSize_out*yIndex
	    grayImg_set[num_total] = grayImages[zStart:zStart+patchZ, xStart:xStart+patchSize, yStart:yStart+patchSize]
	    num_total += 1
	
	yStart = img.shape[1]-patchSize
	for xIndex in range(numSample_axis-1):
	    xStart = patchSize_out*xIndex
	    grayImg_set[num_total] = grayImages[zStart:zStart+patchZ, xStart:xStart+patchSize, yStart:yStart+patchSize]
	    num_total += 1

	xStart = img.shape[0]-patchSize
	yStart = img.shape[1]-patchSize
	grayImg_set[num_total] = grayImages[zStart:zStart+patchZ, xStart:xStart+patchSize, yStart:yStart+patchSize]
	num_total += 1


    for val_ind in range(num_total):
        data_x = grayImg_set[val_ind].astype(np.float32)
        data_x = np.reshape(data_x, [-1, 1, patchZ, patchSize, patchSize])
        im_pred = model.predict(x=data_x, batch_size=1)
	membrane_set[val_ind] = np.reshape(im_pred, (patchZ_out, patchSize_out, patchSize_out))


    num_total = 0
    for zIndex in range(numZ):
        if zIndex == numZ-1:
	    zStart = numProb-patchZ_out
	else:
	    zStart = patchZ_out*zIndex

	for xIndex in range(numSample_axis-1):
	    xStart = patchSize_out*xIndex
	    for yIndex in range(numSample_axis-1):
		yStart = patchSize_out*yIndex
		probImages[zStart:zStart+patchZ_out, xStart:xStart+patchSize_out, yStart:yStart+patchSize_out]=membrane_set[num_total]
		num_total += 1

	xStart = (numSample_axis-1)*patchSize_out
	for yIndex in range(numSample_axis-1):
	    yStart = patchSize_out*yIndex
	    probImages[zStart:zStart+patchZ_out, xStart: , yStart:yStart+patchSize_out]=membrane_set[num_total, :, xStart-img.shape[0]+2*cropSize:, :]
	    num_total += 1
	
	yStart = (numSample_axis-1)*patchSize_out
	for xIndex in range(numSample_axis-1):
	    xStart = patchSize_out*xIndex
	    probImages[zStart:zStart+patchZ_out, xStart:xStart+patchSize_out , yStart:]=membrane_set[num_total, :, :, yStart-img.shape[0]+2*cropSize:]
	    num_total += 1

	xStart = (numSample_axis-1)*patchSize_out
	yStart = (numSample_axis-1)*patchSize_out
	probImages[zStart:zStart+patchZ_out, xStart:, yStart:]=membrane_set[num_total, :, xStart-img.shape[0]+2*cropSize:, yStart-img.shape[0]+2*cropSize:]
	num_total += 1

    if writeImage:
	print 'Store images'
	for imgIndex in range(numProb):
	    scipy.misc.imsave(pathPrefix+"result/prediction_"+str("%04d"%imgIndex)+".tif", probImages[imgIndex])
		
    end_time = time.time()
    total_time = (end_time - start_time)

    print 'Running time: ', total_time / 60.
    print 'finished the prediction'

    if returnValue:
        newMembrane = np.zeros((numProb, (img.shape[0]-2*cropSize)**2))
        newProb_set = np.zeros((numProb, (img.shape[0]-2*cropSize)**2))
        for i in range(numProb):
            newMembrane[i] = crop_image_layer(labelImages[i+csZ,:,:], cropSize).flatten()
            newProb_set[i] = probImages[i].flatten()

	newMembrane=newMembrane.astype(np.int)
        return  newProb_set, newMembrane  
