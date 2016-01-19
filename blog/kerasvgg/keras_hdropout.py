# Hinton's dropout is scaled by retention during forward propagation. Let's not do that....

import theano
import keras

from keras.layers.core import MaskedLayer
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

srng = RandomStreams(seed=np.random.randint(10e6))


class HDropout(MaskedLayer):
    '''
        Hinton's dropout.
    '''
    def __init__(self, p):
        super(HDropout, self).__init__()
        self.p = p

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.p > 0.:
            retain_prob = 1. - self.p
            if train:
                X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            else:
                # X *= retain_prob
                X = X
        return X

    def get_config(self):
        return {"name": self.__class__.__name__,
                "p": self.p}

