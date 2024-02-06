import torch
import torch.nn as nn

from .utils import *

class Layer(object):
    """
    Base class for any layer. Safely convert its parameters and 
    keep track of its input shape.
    """

    def __init__(self, config, in_shape):
        self.config = {k:safe_conversion(v) for k,v in config.items()}
        self.in_shape = in_shape

    def get_module(self):
        raise NotImplementedError
    
    def get_out_shape(self):
        raise NotImplementedError


class ConvolveSpatial(Layer):
    """
    Base class for any layer that uses some sort of 2d convolution
    (conv2d, maxpool2d for now, maybe refactor this) 
    """

    def __init__(self, config, in_shape):

        super(ConvolveSpatial, self).__init__(config, in_shape)
        self.config['stride'] = self.config.get('stride', 1)
        self.config['padding'] = padding_type(self.in_shape[1:], 
                                self.config)

    def get_out_shape(self):
        spatial = self._spatial_transf()
        channel = self._channel_transf() 

        return torch.cat([channel, spatial])

    def _spatial_transf(self):
        return out_conv(self.in_shape[1:], self.config)
        
    def _channel_transf(self):
        raise NotImplementedError


class Dense(Layer):
    """
    Base class for any layer that uses dense transformations
    (i.e. matrix mult), linear and rnn.
    """
    def __init__(self, config, in_shape):

        super(Dense, self).__init__(config, in_shape)
        self.changed_feat = None
        self.out_mod = 1.

    def get_out_shape(self):
        out_shape = self.in_shape.clone()
        out_shape[-1] = self.out_mod*self.config[self.changed_feat]

        return out_shape