# Import Keras Libraries
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

# Import Caffe Libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import caffe
import caffe.io
import collections

# Import Preprocessing Libaries
import karpathy_preprocess

# Import Keras Models
import keras_vgg

########### FUNCTIONALITY FROM CAFFE ###########

# Get the parameters from Caffe, returned in the variable "params"
def get_caffe_params( netname, paramname ):
  # load the model in
  net = caffe.Net(netname, paramname, caffe.TEST)
  params = collections.OrderedDict()
  
  # Read all the parameters into numpy arrays
  for layername in net.params:
    caffelayer = net.params[layername]
    params[layername] = []
    for sublayer in caffelayer:
      params[layername].append( sublayer.data ) 
    print "layer "+layername+" has "+str(len(caffelayer))+" sublayers, shape "+str(params[layername][0].shape)

  return params, net

# A lot of this is taken from Andrej Karpathy's code. Please reference him if you reference us!
def read_image_list( image_list , image_dims=(256,256), crop_dims=(227,227) ):

  input_images = []
  for IMAGE_FILE in open(image_list,'r').read().split():
    input_images.append( caffe.io.resize_image( caffe.io.load_image(IMAGE_FILE), (256,256) ))

  # Scale to standardize input dimensions.
  input_ = np.zeros((len(input_images),
                     image_dims[0],
                     image_dims[1],
                     input_images[0].shape[2]),
                    dtype=np.float32)
  for ix, in_ in enumerate(input_images):
    input_[ix] = caffe.io.resize_image(in_, image_dims)
    
  if oversample:
    # Generate center, corner, and mirrored crops.
    input_ = caffe.io.oversample(input_, crop_dims)
  else:
    # Take center crop.
    center = np.array(self.image_dims) / 2.0
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -self.crop_dims / 2.0,
         self.crop_dims / 2.0
         ])
    input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

  # Classify
  caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                      dtype=np.float32)
  for ix, in_ in enumerate(input_):
    caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
    
  return images

########### FUNCTIONALITY FROM KERAS ###########

# KERAS
def set_keras_params( model, params ):

  weightlayers=[]
  layerindex = 0
  for layer in model.layers:
    if len(layer.get_weights()) > 0:
      weightlayers.append(layerindex)
    layerindex+=1
  print "There are "+str(len(weightlayers))+" layers in the model with weights"

  if len(weightlayers) != len(params):
    print "ERROR: caffe model and specified keras model do not match"
    return model 

  paramkeys = params.keys()

  for i in xrange(0,len(params)):
    layer = model.layers[ weightlayers[i] ]
    weights = params[paramkeys[i]]

    # Dense layers are specified as Input-Output in Keras
    if type(layer) is Dense:
      weights[0] = weights[0].transpose(1,0)
      weights[1] = weights[1]
    # Convolution 2D is specified as flip and then multiply
    elif type(layer) is Convolution2D:
      weights[0] = weights[0].transpose(0,1,2,3)[:,:,::-1,::-1]
    layer.set_weights( weights )
    
  return model

def set_keras_partial( model, params ):

  weightlayers=[]
  layerindex = 0
  for layer in model.layers:
    if len(layer.get_weights()) > 0:
      weightlayers.append(layerindex)
    layerindex+=1
  print "There are "+str(len(weightlayers))+" layers in the model with weights"

  paramkeys = params.keys()

  for i in xrange(0,len(params)):
    if i > (len(weightlayers)-1):
      break
    layer = model.layers[ weightlayers[i] ]
    weights = params[paramkeys[i]]
    if type(layer) is Dense:
      weights[0] = weights[0].transpose(1,0)
      weights[1] = weights[1]
    else:
      weights[0] = weights[0].transpose(0,1,2,3)[:,:,::-1,::-1]
    layer.set_weights( weights )
    print "Finished caffe("+str(i)+ " corresponding to keras layer "+str(weightlayers[i] )

  return model


# Transfer caffe network to keras
def caffe2keras( caffemodel, caffeparams, kerasmodel ):

  params,net = get_caffe_params( caffemodel, caffeparams )
  kerasmodel = set_keras_params(kerasmodel, params)
  kerasmodel.compile(loss='categorical_crossentropy', optimizer='sgd')

  print "Finished compiling categoral crossentropy on VGG network."

  return kerasmodel,net


# Transfer caffe network to keras
def caffe2keras_partial( caffemodel, caffeparams, kerasmodel ):

  params,net = get_caffe_params( caffemodel, caffeparams )
  kerasmodel = set_keras_partial(kerasmodel, params)
  kerasmodel.compile(loss='categorical_crossentropy', optimizer='sgd')

  print "Finished compiling categoral crossentropy on VGG network."

  return kerasmodel,net
 
# VGG Net 16 Layers
def transfer_vgg():
   
  import keras_vgg
  netname='VGG_ILSVRC_16_layers_deploy.prototxt'
  paramname='VGG_ILSVRC_16_layers.caffemodel'
  
  params,net = get_caffe_params( netname, paramname )
  reload(keras_vgg)
  model = keras_vgg.vggmodel()
  model = set_keras_params(model, params)
  model.compile(loss='categorical_crossentropy', optimizer='sgd')

  print "Finished compiling categoral crossentropy on VGG network."

  return model,net



