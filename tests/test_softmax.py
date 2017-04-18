import numpy as np
import nose
from random import randrange



def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
  """
  sample a few random elements and only return numerical
  in this dimensions.
  """

  for i in xrange(num_checks):
    ix = tuple([randrange(m) for m in x.shape])

    oldval = x[ix]
    x[ix] = oldval + h # increment by h
    fxph = f(x) # evaluate f(x + h)
    x[ix] = oldval - h # increment by h
    fxmh = f(x) # evaluate f(x - h)
    x[ix] = oldval # reset

    grad_numerical = (fxph - fxmh) / (2 * h)
    grad_analytic = analytic_grad[ix]
    rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
    yield (grad_numerical, grad_analytic, rel_error)

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """

  fx = f(x)  # evaluate function value at original point
  grad = np.zeros_like(x)
  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    oldval = x[ix]
    x[ix] = oldval + h  # increment by h
    fxph = f(x)  # evalute f(x + h)
    x[ix] = oldval - h
    fxmh = f(x)  # evaluate f(x - h)
    x[ix] = oldval  # restore

    # compute the partial derivative with centered formula
    grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
    if verbose:
      print ix, grad[ix]
    it.iternext()  # step to next dimension

  return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
  """
  Evaluate a numeric gradient for a function that accepts a numpy
  array and returns a numpy array.
  """
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index

    oldval = x[ix]
    x[ix] = oldval + h
    pos = f(x).copy()
    x[ix] = oldval - h
    neg = f(x).copy()
    x[ix] = oldval

    grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    it.iternext()
  return grad


def test_loss_naive():
  from code.classifiers.softmax import softmax_loss_naive
  import time

  # Generate a random softmax weight matrix and use it to compute the loss.
  np.random.seed(1)
  W = np.random.randn(3073, 10) * 0.00001
  X = np.random.randn(500, 3073)
  y = np.random.randint(0, 10, size=(500,))

  loss, grad = softmax_loss_naive(W, X, y, 0.0)

  # As a rough sanity check, our loss should be something close to -log(0.1).
  assert np.abs(loss - 2.3) < 0.1, 'Problem with softmax naive loss. Loss is %f, but should be 2.3' % loss


def test_naive_gradients():
  from code.classifiers.softmax import softmax_loss_naive
  np.random.seed(1)
  W = np.random.randn(3073, 10) * 0.00001
  X = np.random.randn(500, 3073)
  y = np.random.randint(0, 10, size=(500,))

  loss, grad = softmax_loss_naive(W, X, y, 0.0)

  f = lambda w: softmax_loss_naive(w, X, y, 0.0)[0]
  for numerical, analytical, error in grad_check_sparse(f, W, grad, 10):
    assert error < 5e-7, 'Problem with naive gradients without regularization %s, %s error: %s' % (numerical, analytical, error)

  loss, grad = softmax_loss_naive(W, X, y, 1e2)
  f = lambda w: softmax_loss_naive(w, X, y, 1e2)[0]
  for numerical, analytical, error in grad_check_sparse(f, W, grad, 10):
    assert error < 5e-7, 'Problem with naive gradients with regularization %s, %s error: %s' % (numerical, analytical, error)


def test_loss_vectorized():
  from code.classifiers.softmax import softmax_loss_naive
  np.random.seed(1)
  W = np.random.randn(3073, 10) * 0.00001
  X = np.random.randn(500, 3073)
  y = np.random.randint(0, 10, size=(500,))
  loss_naive, grad_naive = softmax_loss_naive(W, X, y, 0.00001)

  from code.classifiers.softmax import softmax_loss_vectorized
  loss_vectorized, _ = softmax_loss_vectorized(W, X, y, 0.00001)

  # The losses should match but your vectorized implementation should be much faster.
  assert np.abs(loss_naive - loss_vectorized) < 1e-10, 'Problem with vectorized loss. Loss (%f) is different from naive (%f)' %\
                                                       (loss_naive, loss_vectorized)


def test_gradient_vectorized():
  from code.classifiers.softmax import softmax_loss_vectorized as softmax_loss_vectorized
  from code.classifiers.softmax import softmax_loss_naive
  np.random.seed(1)
  W = np.random.randn(3073, 10) * 0.00001
  X = np.random.randn(500, 3073)
  y = np.random.randint(0, 10, size=(500,))

  loss_naive, grad_naive = softmax_loss_naive(W, X, y, 0.0000)

  loss, grad = softmax_loss_vectorized(W, X, y, 0.0)

  f = lambda w: softmax_loss_vectorized(w, X, y, 0.0)[0]
  for numerical, analytical, error in grad_check_sparse(f, W, grad, 10):
    assert error < 5e-7, 'Problem with naive gradients without regularization %s, %s error: %s' % (numerical, analytical, error)

  loss, grad = softmax_loss_vectorized(W, X, y, 1e2)
  f = lambda w: softmax_loss_vectorized(w, X, y, 1e2)[0]
  for numerical, analytical, error in grad_check_sparse(f, W, grad, 10):
    assert error < 5e-7, 'Problem with naive gradients with regularization %s, %s error: %s' % (numerical, analytical, error)


def test_sgd():
  from code.data_utils import get_CIFAR10_data
  from code.classifiers.linear_classifier import Softmax
  softmax = Softmax()
  data = get_CIFAR10_data()
  X_train, y_train = data['X_train'], data['y_train']
  X_val, y_val = data['X_val'], data['y_val']

  np.random.seed(1)
  loss_hist = softmax.train(X_train.reshape((X_train.shape[0], -1)), y_train, learning_rate=1e-7, reg=5e4,
                            num_iters=1500, verbose=False)
  assert loss_hist[-1] < 2.2, 'Problem with softmax train sgd, loss: %s' % loss_hist[-1]
  y_val_pred = softmax.predict(X_val.reshape((X_val.shape[0], -1)))
  val_acc = np.mean(y_val == y_val_pred)
  assert val_acc > 0.28, 'Problem with softmax train sgd, val accuracy %f' % val_acc