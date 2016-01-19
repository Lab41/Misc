from caffekeras_util import caffe2keras_partial
import keras_vgg
import karpathy_preprocess
from keras_hdropout import HDropout
from time import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import numpy as np
import caffe.io

def vggnet(layername='softmax'):

    # define the model
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    if layername is 'conv1':
        return model
    
    # conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> pool2
    model.add(Convolution2D(128, 64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    if layername is 'conv2':
        return model

    # conv3_1 -> relu3_1 -> conv3_2 -> relu3_2 -> conv3_3 -> relu3_3 -> pool3
    model.add(Convolution2D(256, 128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    if layername is 'conv3':
        return model
    
    # conv4_1 -> relu4_1 -> conv4_2 -> relu4_2 -> conv4_3 -> relu4_3 -> pool4
    model.add(Convolution2D(512, 256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    if layername is 'conv4':
        return model
    
    # conv5_1 -> relu5_1 -> conv5_2 -> relu5_2 -> conv5_3 -> relu5_3 -> pool5
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    if layername is 'conv5':
        return model
    
    # Flatten
    model.add(Flatten())

    # fc6
    model.add(Dense(25088,4096))
    model.add(Activation('relu'))
    model.add(HDropout(0.5))
    if layername is 'fc6':
        return model
    
    # fc7
    model.add(Dense(4096,4096))
    model.add(Activation('relu'))
    model.add(HDropout(0.5))
    if layername is 'fc7':
        return model

    # fc8 & softmax
    model.add(Dense(4096, 1000))
    if layername is 'fc8':
        return model
    
    model.add(Activation('softmax'))

    return model

def load_images( imagenames, imagebuffer = None):

    timestart = time()

    if not imagebuffer:
        imagebuffer = []
        for IMAGE_FILE in imagenames: 
            imagebuffer.append( caffe.io.resize_image( caffe.io.load_image(IMAGE_FILE), (256,256) ))
    elif len(imagebuffer) < len(imagenames):
        print "ERROR: Image buffer is not of the correct size"
        return imagebuffer
    else:
        # Only zero pad if necessary
        if len(imagebuffer) > len(imagenames):
            for imagepad in imagebuffer:
                imagepad[...]=np.zeros((3,224,224))
        for imageindex in xrange(0,len(imagenames)):
            imagebuffer[imageindex][...] = karpathy_preprocess.preprocess_image( 
                caffe.io.resize_image( 
                    caffe.io.load_image(imagenames[imageindex]), (256,256)
                    )
                )

    timestop = time()
    print "Time to load "+str(len(imagenames))+ " images: "+str(timestop - timestart)

    return imagebuffer

def load_vgg(layername='fc7'):
    timestart = time()
    model,net = caffe2keras_partial('VGG_ILSVRC_16_layers_deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', vggnet(layername))
    timestop = time()

    print "Time to load both caffe models and keras model is "+str( timestop - timestart )

    return model, net

def layershape(layername):

    if layername is 'fc7' or layername is 'fc6':
        return 4096
    elif layername is 'conv':
        return "not supported"
    else:
        return 1000

def classify_images( filename='testvgg.txt', batchsize = 10, layername = 'fc7' ):

    # Load the model
    timestart = time()
    model, net = load_vgg(layername)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    timestop = time()
    print "Finished loading and converting model in "+str(timestop-timestart)+" seconds"

    # Get the file names
    imagenames = open(filename, 'r').read().split()
    labels = open('/p/lscratche/brainusr/datasets/ILSVRC2012/labels/labels.txt').read().split()

    # Allocate space for output
    predictions = np.zeros( (len(imagenames), layershape(layername) ))

    # Iterate through and classify
    input_buffer = [ np.zeros((3,224,224)) for i in xrange(0, batchsize) ]
    for imagebatch in xrange( 0, len(imagenames), min(len(imagenames),batchsize)):

        maxbatch = min(imagebatch+batchsize, len(imagenames))
        load_images(imagenames[imagebatch:maxbatch], input_buffer)      
        predictions[imagebatch:maxbatch,:] = model.predict( np.float32(np.asarray(input_buffer[0:maxbatch-imagebatch])) )
    
    return predictions 




    

