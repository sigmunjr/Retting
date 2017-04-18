import numpy as np

def init_toy_model(input_size, hidden_size, num_classes):
  from code.classifiers.neural_net import TwoLayerNet
  np.random.seed(0)
  return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data(num_inputs, input_size):
  np.random.seed(1)
  X = 10 * np.random.randn(num_inputs, input_size)
  y = np.array([0, 1, 2, 2, 1])
  return X, y


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_forward_pass():
  input_size = 4
  hidden_size = 10
  num_classes = 3
  num_inputs = 5

  net = init_toy_model(input_size, hidden_size, num_classes)
  X, y = init_toy_data(num_inputs, input_size)
  scores = net.loss(X)

  correct_scores = np.asarray([
    [-0.81233741, -1.27654624, -0.70335995],
    [-0.17129677, -1.18803311, -0.47310444],
    [-0.51590475, -1.01354314, -0.8504215],
    [-0.15419291, -0.48629638, -0.52901952],
    [-0.00618733, -0.12435261, -0.15226949]])

  assert np.abs(scores - correct_scores).sum() < 1e-7, 'Problems with scores in forward pass. Error: %f' % np.abs(scores - correct_scores).sum()
  loss, _ = net.loss(X, y, reg=0.1)
  correct_loss = 1.30378789133

  assert np.sum(np.abs(loss - correct_loss))<1e-12, 'Problem with loss in forward pass. Loss is %f should be %f' % (loss, correct_loss)


def test_backward_pass():
  from code.gradient_check import eval_numerical_gradient
  input_size = 4
  hidden_size = 10
  num_classes = 3
  num_inputs = 5

  net = init_toy_model(input_size, hidden_size, num_classes)
  X, y = init_toy_data(num_inputs, input_size)

  loss, grads = net.loss(X, y, reg=0.1)

  # these should all be less than 1e-8 or
  for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.1)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))