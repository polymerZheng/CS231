import numpy as np
from random import shuffle
from past.builtins import xrange

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
  y_soft=np.zeros((X.shape[0],W.shape[1]))
  y_true=np.zeros_like(y_soft)
  y_true[np.arange(X.shape[0]),y]=1
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    for j in range(W.shape[1]):
      y_soft[i][j]=X[i].dot(W[:,j])
  C=np.amax(y_soft,axis=1,keepdims=True)
  y_soft-=C
  y_soft=np.exp(y_soft)
  y_sum=np.sum(y_soft,axis=1,keepdims=True)
  y_soft/=y_sum
  y_mat=y_soft[np.arange(X.shape[0]),y]
  loss=np.sum(-np.log(y_mat))/X.shape[0]
  loss+=0.5*np.sum(W*W)*reg
  dW=np.matmul(X.T,-(y_true-y_soft))/X.shape[0]
  dW+=reg*W

  
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
  y_soft=np.zeros((X.shape[0],W.shape[1]))
  y_true=np.zeros_like(y_soft)
  y_true[np.arange(X.shape[0]),y]=1
  y_soft=np.matmul(X,W)
  C=np.amax(y_soft,axis=1,keepdims=True)
  y_soft-=C
  y_soft=np.exp(y_soft)
  y_sum=np.sum(y_soft,axis=1,keepdims=True)
  y_soft/=y_sum
  y_mat=y_soft[np.arange(X.shape[0]),y]
  loss=np.sum(-np.log(y_mat))/X.shape[0]
  loss+=0.5*np.sum(W*W)*reg
  dW=np.matmul(X.T,-(y_true-y_soft))/X.shape[0]
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

