import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, dim = X.shape
  num_class = W.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    correct_class_score = scores[y[i]]
    exp_scores = np.exp(scores)
    loss -= correct_class_score
    expsum = np.sum(exp_scores)
    loss += np.log(expsum)
    for c in range(num_class):
      dW[:, c] += X[i] * (exp_scores[c] / expsum)
      if c == y[i]:
        dW[:, c] -= X[i]
    

  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, dim = X.shape
  num_class = W.shape[1]

  scores = X.dot(W)
  scores -= np.max(scores)
  exp_scores = np.exp(scores)
  correct_class_scores = scores[np.arange(num_train), y]
  loss -= np.sum(correct_class_scores)
  exp_sums = np.sum(exp_scores, axis = 1)
  loss += np.sum(np.log(exp_sums))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  positive_contributions = exp_scores / np.tile(exp_sums.reshape((num_train, 1)), (1, num_class))
  mask = np.zeros(scores.shape)
  mask[np.arange(num_train), y] = 1
  negative_contributions = mask
  dW = X.T.dot(positive_contributions - negative_contributions)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

