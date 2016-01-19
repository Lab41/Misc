from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

def vggmodel():

    # define the model
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> pool2
    model.add(Convolution2D(128, 64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))

    # conv3_1 -> relu3_1 -> conv3_2 -> relu3_2 -> conv3_3 -> relu3_3 -> pool3
    model.add(Convolution2D(256, 128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # conv4_1 -> relu4_1 -> conv4_2 -> relu4_2 -> conv4_3 -> relu4_3 -> pool4
    model.add(Convolution2D(512, 256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # conv5_1 -> relu5_1 -> conv5_2 -> relu5_2 -> conv5_3 -> relu5_3 -> pool5
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    
    # Flatten
    model.add(Flatten())

    # fc6
    model.add(Dense(25088,4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # fc7
    model.add(Dense(4096,4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # fc8
    model.add(Dense(4096,1000))
    model.add(Activation('softmax'))

    return model

def keras_vgg_truncate():

    # define the model
    model = Sequential()

    if True:
      model.add(Convolution2D(64, 3, 3, 3, border_mode='same', activation='relu'))
      model.add(Convolution2D(64, 64, 3, 3, border_mode='same', activation='relu'))
      model.add(MaxPooling2D(poolsize=(2, 2)))

    if True:
      # conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> pool2
      model.add(Convolution2D(128, 64, 3, 3, border_mode='same', activation='relu'))
      model.add(Convolution2D(128, 128, 3, 3, border_mode='same', activation='relu'))
      model.add(MaxPooling2D(poolsize=(2, 2)))

    if True:
      # conv3_1 -> relu3_1 -> conv3_2 -> relu3_2 -> conv3_3 -> relu3_3 -> pool3
      model.add(Convolution2D(256, 128, 3, 3, border_mode='same', activation='relu'))
      model.add(Convolution2D(256, 256, 3, 3, border_mode='same', activation='relu'))
      model.add(Convolution2D(256, 256, 3, 3, border_mode='same', activation='relu'))
      model.add(MaxPooling2D(poolsize=(2, 2)))

    if True:
      # conv4_1 -> relu4_1 -> conv4_2 -> relu4_2 -> conv4_3 -> relu4_3 -> pool4
      model.add(Convolution2D(512, 256, 3, 3, border_mode='same', activation='relu'))
      model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model.add(MaxPooling2D(poolsize=(2, 2)))

    if True:
      # conv5_1 -> relu5_1 -> conv5_2 -> relu5_2 -> conv5_3 -> relu5_3 -> pool5
      model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model.add(MaxPooling2D(poolsize=(2, 2)))

    if True:
      # Flatten -> fc6
      model.add(Flatten())
      model.add(Dense(25088,4096))
      model.add(Activation('relu'))
      model.add(Dropout(0.5))

    if False:
      # fc7
      model.add(Dense(4096,4096))
      model.add(Activation('relu'))
      model.add(Dropout(0.5))

    if False:
      # fc8 and softmax
      model.add(Dense(4096,1000))
      model.add(Activation('softmax'))

    return model

def keras_vgg_segmented( params, modelSeq = None ):

    LoadWeights = True

    if not modelSeq:
        model1 = Sequential()
        model2 = Sequential()
        model3 = Sequential()
        model4 = Sequential()
        model5 = Sequential()
        model6 = Sequential()
        model7 = Sequential()
        model8 = Sequential()
    else:
        model1 = modelSeq[0]
        model2 = modelSeq[1]
        model3 = modelSeq[2]
        model4 = modelSeq[3]
        model5 = modelSeq[4]
        model6 = modelSeq[5]
        model7 = modelSeq[6]
        model8 = modelSeq[7]

    if True and not modelSeq:
      model1.add(Convolution2D(64, 3, 3, 3, border_mode='same', activation='relu'))
      model1.add(Convolution2D(64, 64, 3, 3, border_mode='same', activation='relu'))
      model1.add(MaxPooling2D(poolsize=(2, 2)))

    if True and LoadWeights:
      params[ 'conv1_1' ][0][...]=params['conv1_1'][0][:,:,::-1,::-1]
      model1.layers[0].set_weights( params['conv1_1'] )
      params[ 'conv1_2' ][0][...]=params['conv1_2'][0][:,:,::-1,::-1]
      model1.layers[1].set_weights( params['conv1_2'] )

    if True and not modelSeq:
      # conv2_1 -> relu2_1 -> conv2_2 -> relu2_2 -> pool2
      model2.add(Convolution2D(128, 64, 3, 3, border_mode='same', activation='relu'))
      model2.add(Convolution2D(128, 128, 3, 3, border_mode='same', activation='relu'))
      model2.add(MaxPooling2D(poolsize=(2, 2)))

    if True and LoadWeights:
      params[ 'conv2_1' ][0][...]=params['conv2_1'][0][:,:,::-1,::-1]
      model2.layers[0].set_weights( params['conv2_1'] )
      params[ 'conv2_2' ][0][...]=params['conv2_2'][0][:,:,::-1,::-1]
      model2.layers[1].set_weights( params['conv2_2'] )

    if True and not modelSeq:
      # conv3_1 -> relu3_1 -> conv3_2 -> relu3_2 -> conv3_3 -> relu3_3 -> pool3
      model3.add(Convolution2D(256, 128, 3, 3, border_mode='same', activation='relu'))
      model3.add(Convolution2D(256, 256, 3, 3, border_mode='same', activation='relu'))
      model3.add(Convolution2D(256, 256, 3, 3, border_mode='same', activation='relu'))
      model3.add(MaxPooling2D(poolsize=(2, 2)))

    if True and LoadWeights:
      params[ 'conv3_1' ][0][...]=params['conv3_1'][0][:,:,::-1,::-1]
      model3.layers[0].set_weights( params['conv3_1'] )
      params[ 'conv3_2' ][0][...]=params['conv3_2'][0][:,:,::-1,::-1]
      model3.layers[1].set_weights( params['conv3_2'] )
      params[ 'conv3_3' ][0][...]=params['conv3_3'][0][:,:,::-1,::-1]
      model3.layers[2].set_weights( params['conv3_3'] )

    if True and not modelSeq:
      # conv4_1 -> relu4_1 -> conv4_2 -> relu4_2 -> conv4_3 -> relu4_3 -> pool4
      model4.add(Convolution2D(512, 256, 3, 3, border_mode='same', activation='relu'))
      model4.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model4.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model4.add(MaxPooling2D(poolsize=(2, 2)))

    if True and LoadWeights:
      params[ 'conv4_1' ][0][...]=params['conv4_1'][0][:,:,::-1,::-1]
      model4.layers[0].set_weights( params['conv4_1'] )
      params[ 'conv4_2' ][0][...]=params['conv4_2'][0][:,:,::-1,::-1]
      model4.layers[1].set_weights( params['conv4_2'] )
      params[ 'conv4_3' ][0][...]=params['conv4_3'][0][:,:,::-1,::-1]
      model4.layers[2].set_weights( params['conv4_3'] )


    if True and not modelSeq:
      # conv5_1 -> relu5_1 -> conv5_2 -> relu5_2 -> conv5_3 -> relu5_3 -> pool5
      model5.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model5.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model5.add(Convolution2D(512, 512, 3, 3, border_mode='same', activation='relu'))
      model5.add(MaxPooling2D(poolsize=(2, 2)))
      model5.add(Flatten())

    if True and LoadWeights:
      params[ 'conv5_1' ][0][...]=params['conv5_1'][0][:,:,::-1,::-1]
      model5.layers[0].set_weights( params['conv5_1'] )
      params[ 'conv5_2' ][0][...]=params['conv5_2'][0][:,:,::-1,::-1]
      model5.layers[1].set_weights( params['conv5_2'] )
      params[ 'conv5_3' ][0][...]=params['conv5_3'][0][:,:,::-1,::-1]
      model5.layers[2].set_weights( params['conv5_3']  )

    if True and not modelSeq:
      # Flatten -> fc6
      model6.add(Dense(25088,4096))
      model6.add(Activation('relu'))
      model6.add(Dropout(0.5))

    if True and LoadWeights:
      model6.layers[0].set_weights( [2.0*params['fc6'][0].transpose(1,0), 2.0*params['fc6'][1]] )

    if True and not modelSeq:
      # fc7
      model7.add(Dense(4096,4096))
      model7.add(Activation('relu'))
      model7.add(Dropout(0.5))

    if True and LoadWeights:
      model7.layers[0].set_weights( [2.0*params['fc7'][0].transpose(1,0), 2.0*params['fc7'][1]] )

    if True and not modelSeq:
      # fc8 and softmax
      model8.add(Dense(4096,1000))
      # model8.add(Activation('softmax'))

    if True and LoadWeights:
      model8.layers[0].set_weights( [params['fc8'][0].transpose(1,0), params['fc8'][1]] )

    return model1, model2, model3, model4, model5, model6, model7, model8

