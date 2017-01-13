import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  sum_h = x.dot(Wx) + prev_h.dot(Wh) + b
  next_h = np.tanh(sum_h)
  cache = (x, prev_h, Wx, Wh, b, sum_h)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  x, prev_h, Wx, Wh, b, sum_h = cache
  z = sum_h
  dz = 1 - np.square(np.tanh(z))
  dz *= dnext_h
  dx = dz.dot(Wx.T)
  dprev_h = dz.dot(Wh.T)
  dWx = dz.T.dot(x).T   ## remind this exactly
  dWh = dz.T.dot(prev_h).T
  db = dz.sum(axis=0)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  _, H = h0.shape
  h = np.zeros((N, T, H))
  caches = []
  h[:,0,:], cache = rnn_step_forward(x[:,0,:], h0, Wx, Wh, b)
  caches.append(cache)
  for t in range(1, T):
    h[:,t,:], cache = rnn_step_forward(x[:,t,:], h[:,t-1,:], Wx, Wh, b)
    caches.append(cache)
  cache = caches
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  N, T, H = dh.shape
  D = cache[0][0].shape[1]
  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, H))
  dWh = np.zeros((H, H))
  db = np.zeros(H)
  dprev_h = np.zeros((N,H))
  for t in reversed(range(T)):
    c = cache[t]
    dx_, dprev_h, dWx_, dWh_, db_ = rnn_step_backward(dh[:,t,:]+dprev_h, c)
    dx[:,t,:] = dx_
    dWx += dWx_
    dWh += dWh_
    db += db_
    #dh = dprev_h
  dh0 = dprev_h

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  out = W[x]
  cache = (x, W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x, W = cache
  T = dout.shape[1]
  dW = np.zeros(W.shape)
  np.add.at(dW, x, dout)
  #for t in range(T):
  #  np.add.at(dW, x[:,t], dout[:,t])
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  a = x.dot(Wx) + prev_h.dot(Wh) + b
  N, H = prev_h.shape
  sigm = sigmoid(a[:,:3*H])
  i, f, o = sigm[:,:H], sigm[:,H:2*H], sigm[:,2*H:3*H]
  g = np.tanh(a[:,3*H:])
  next_c = f * prev_c + i * g
  next_h = o * np.tanh(next_c)
  cache = (x, prev_h, prev_c, next_c, next_h, Wx, Wh, b, a, f, i, o, g)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  x, prev_h, prev_c, c, h, Wx, Wh, b, a, f, i, o, g = cache
  #dh_dc
  dh_dc = o * (1 - np.square(np.tanh(c)))
  dc_dprevc = f
  #dprev_c = dc_dprevc * dnext_c + dh_dc * dnext_h
  dL_dc = dnext_h * dh_dc + dnext_c
  dprev_c = dL_dc * dc_dprevc
  ##
  dc_di = g
  dc_df = prev_c
  dh_do = np.tanh(c)
  dc_dg = i
  ##
  dL_di = dL_dc * dc_di
  dL_df = dL_dc * dc_df
  dL_do = dnext_h * dh_do
  dL_dg = dL_dc * dc_dg
  #
  di_da = i * (1-i)
  df_da = f * (1-f)
  do_da = o * (1-o)
  dg_da = 1 - np.square(g)
  #
  dL_da = np.hstack((dL_di*di_da, dL_df*df_da, dL_do*do_da, dL_dg*dg_da))

  dx = dL_da.dot(Wx.T)
  dprev_h = dL_da.dot(Wh.T)
  dWx = dL_da.T.dot(x).T
  dWh = dL_da.T.dot(prev_h).T
  db = dL_da.sum(axis=0)


  # dx = np.zeros(x.shape)
  # dprev_h = np.zeros(h.shape)
  # dWx = np.zeros(Wx.shape)
  # dWh = np.zeros(Wh.shape)
  # db = np.zeros(b.shape)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  _, H = h0.shape
  h = np.zeros((N, T, H))
  caches = []
  #lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):; next_h, next_c, cache
  c = np.zeros((N, H))
  h[:,0,:], c, cache = lstm_step_forward(x[:,0,:], h0, c, Wx, Wh, b)
  caches.append(cache)
  for t in range(1, T):
    h[:,t,:], c, cache = lstm_step_forward(x[:,t,:], h[:,t-1,:], c, Wx, Wh, b)
    caches.append(cache)
  cache = caches
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  # dx, dprev_h, dprev_c, dWx, dWh, db <-lstm_step_backward(dnext_h, dnext_c, cache):
  N, T, H = dh.shape
  D = cache[0][0].shape[1]
  dx = np.zeros((N, T, D))
  dh0 = np.zeros((N, H))
  dWx = np.zeros((D, 4*H))
  dWh = np.zeros((H, 4*H))
  db = np.zeros(4*H)
  dprev_h = np.zeros((N,H))
  dprev_c = np.zeros((N,H))
  for t in reversed(range(T)):
    cache_ = cache[t]
    dx_, dprev_h, dprev_c, dWx_, dWh_, db_ = lstm_step_backward(dh[:,t,:]+dprev_h, dprev_c, cache_)
    dx[:,t,:] = dx_
    dWx += dWx_
    dWh += dWh_
    db += db_
    #dh = dprev_h
  dh0 = dprev_h
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

if __name__=='__main__':
  from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

  N, D, H = 4, 5, 6
  x = np.random.randn(N, D)
  h = np.random.randn(N, H)
  Wx = np.random.randn(D, H)
  Wh = np.random.randn(H, H)
  b = np.random.randn(H)

  out, cache = rnn_step_forward(x, h, Wx, Wh, b)

  dnext_h = np.random.randn(*out.shape)

  # fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]
  # fh = lambda prev_h: rnn_step_forward(x, h, Wx, Wh, b)[0]
  # fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]
  # fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]
  # fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]

  # dx_num = eval_numerical_gradient_array(fx, x, dnext_h)
  # dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)
  # dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)
  # dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)
  # db_num = eval_numerical_gradient_array(fb, b, dnext_h)
  # dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)

  # print 'dx error: ', rel_error(dx_num, dx)
  # print 'dprev_h error: ', rel_error(dprev_h_num, dprev_h)
  # print 'dWx error: ', rel_error(dWx_num, dWx)
  # print 'dWh error: ', rel_error(dWh_num, dWh)
  # print 'db error: ', rel_error(db_num, db)




  N, D, T, H = 4, 5, 1, 6

  x = x[:,np.newaxis,:]
  h0 = h
  #x = np.random.randn(N, T, D)
  #h0 = np.random.randn(N, H)
  #Wx = np.random.randn(D, H)
  #Wh = np.random.randn(H, H)
  #b = np.random.randn(H)
  out1, cache1 = out, cache
  print cache1[1]
  out, cache = rnn_forward(x, h0, Wx, Wh, b)
  out2, cache2 = out, cache[0]
  for i in range(len(cache1)):
    print cache1[i]==cache2[i]

  #dout = np.random.randn(*out.shape)
  dout = dnext_h[:,np.newaxis,:]

  dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)

  fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]
  fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]
  fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]
  fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]
  fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]

  dx_num = eval_numerical_gradient_array(fx, x, dout)
  dh0_num = eval_numerical_gradient_array(fh0, h0, dout)
  dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)
  dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)
  db_num = eval_numerical_gradient_array(fb, b, dout)
  print dWh
  print dWh_num
  print 'dx error: ', rel_error(dx_num, dx)
  print 'dh0 error: ', rel_error(dh0_num, dh0)
  print 'dWx error: ', rel_error(dWx_num, dWx)
  print 'dWh error: ', rel_error(dWh_num, dWh)
  print 'db error: ', rel_error(db_num, db)

