import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    #conv_outsize = num_filters*(W-filter_size+1)*(H-filter_size)+1
    conv_outsize = num_filters*H/2*W/2 # padding + maxpool
    self.params['W2'] = weight_scale * np.random.randn(conv_outsize, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    X, cache1 = conv_relu_forward(X, W1, b1, conv_param)
    X, cache1p = max_pool_forward_fast(X, pool_param)
    X, cache2 = affine_relu_forward(X, W2, b2)
    X, cache3 = affine_forward(X, W3, b3)
    scores = X

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    dx, dw3, db3 = affine_backward(dx, cache3)
    dx, dw2, db2 = affine_relu_backward(dx, cache2)
    dx = max_pool_backward_fast(dx, cache1p)
    dx, dw1, db1 = conv_relu_backward(dx, cache1)

    loss += 0.5*self.reg*(np.sum(np.square(W1)) +np.sum(np.square(W2)) + np.sum(np.square(W3)))
    grads['W1'] = dw1 + self.reg*W1
    grads['W2'] = dw2 + self.reg*W2
    grads['W3'] = dw3 + self.reg*W3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
def init_conv(F, C, H, weight_scale):
    return weight_scale * np.random.randn(F, C, H, H), np.zeros(F)
  
'''2 convs with relu, max pool, 2 convs with relu, maxpool, conv, relu, affine batchnorm relu affine'''
class CustomNet:

  def __init__(self, input_dim=(3, 32, 32), num_filters=[16, 16, 32, 32, 64], filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    C, H, W = input_dim

    self.params['W1a'], self.params['b1a'] = init_conv(num_filters[0], C, filter_size, weight_scale)
    self.params['W1b'], self.params['b1b'] = init_conv(num_filters[1], num_filters[0], filter_size, weight_scale)

    self.params['W2a'], self.params['b2a'] = init_conv(num_filters[2], num_filters[1], filter_size, weight_scale)
    self.params['W2b'], self.params['b2b'] = init_conv(num_filters[3], num_filters[2], filter_size, weight_scale)

    self.params['W3'], self.params['b3'] = init_conv(num_filters[4], num_filters[3], filter_size, weight_scale)

    self.params['W4'] = weight_scale*np.random.randn(num_filters[1]*H/2*W/2, hidden_dim)
    self.params['b4'] = np.zeros(hidden_dim)
    self.params['gamma1'], self.params['beta1'] = np.ones(hidden_dim), np.zeros(hidden_dim)
    self.params['W5'] = weight_scale*np.random.randn(hidden_dim, num_classes)
    self.params['b5'] = np.zeros(num_classes)

    self.bn_param = {}

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'
    bn_param = self.bn_param
    bn_param['mode'] = mode
    W1a, b1a = self.params['W1a'], self.params['b1a']
    W1b, b1b = self.params['W1b'], self.params['b1b']
    W2a, b2a = self.params['W2a'], self.params['b2a']
    W2b, b2b = self.params['W2b'], self.params['b2b']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1a.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    X, cache1a = conv_relu_forward(X, W1a, b1a, conv_param)
    X, cache1b = conv_relu_forward(X, W1b, b1b, conv_param)
    X, cache1p = max_pool_forward_fast(X, pool_param)

    X, cache2a = conv_relu_forward(X, W2a, b2a, conv_param)
    X, cache2b = conv_relu_forward(X, W2b, b2b, conv_param)
    X, cache2p = max_pool_forward_fast(X, pool_param)

    X, cache3 = conv_relu_forward(X, W3, b3, conv_param)
    
    X, cache4 = affine_batchnorm_relu_forward(X, W4, b4, gamma1, beta1, bn_param)
    X, cache5 = affine_forward(X, W5, b5)
    scores = X
    
    if y is None:
      return scores
    
    loss, grads = 0, {}

    loss, dx = softmax_loss(scores, y)
    dx, dw5, db5 = affine_backward(dx, cache5)
    #dx, dw4, db4 = affine_relu_backward(dx, cache4)
    dx, dw4, db4, dgamma1, dbeta1 = affine_batchnorm_relu_backward(dx, cache4)
    dx, dw3, db3 = conv_relu_backward(dx, cache3)
    dx = max_pool_backward_fast(dx, cache2p)
    dx, dw2b, db2b = conv_relu_backward(dx, cache2b)
    dx, dw2a, db2a = conv_relu_backward(dx, cache2a)
    dx = max_pool_backward_fast(dx, cache1p)
    dx, dw1b, db1b = conv_relu_backward(dx, cache1b)
    dx, dw1a, db1a = conv_relu_backward(dx, cache1a)

    loss += 0.5*self.reg*(np.sum(np.square(W1a)) + np.sum(np.square(W1b)) + 
        np.sum(np.square(W2a)) + np.sum(np.square(W2b)) + np.sum(np.square(W3)) + 
        np.sum(np.square(W4) + np.sum(np.square(W5))))
    grads['W1a'] = dw1a + self.reg*W1a
    grads['W1b'] = dw1b + self.reg*W1b
    grads['W2a'] = dw2a + self.reg*W2a
    grads['W2b'] = dw2b + self.reg*W2b
    grads['W3'] = dw3 + self.reg*W3
    grads['W4'] = dw4 + self.reg*W4
    grads['W5'] = dw5 + self.reg*W5
    grads['b1a'] = db1a
    grads['b1b'] = db1b
    grads['b2a'] = db2a
    grads['b2b'] = db2b
    grads['b3'] = db3
    grads['b4'] = db4
    grads['b5'] = db5
    grads['beta1'] = dbeta1
    grads['gamma1'] = dgamma1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
