import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
old_v = tf.compat.v1.logging.get_verbosity()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from sklearn.manifold import TSNE
from tensorflow.python.framework import ops
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def get_session(gpu_fraction=0.5):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
 
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


class FlipGradientBuilder(object):
    """
    Gradient Reversal Layer (GRL)
    
    During the forward propagation, GRL acts as an identity transform
    
    During the backpropagation though, GRL takes the gradient from the subsequent level,
    multiplies it by -λ(learning rate) and pass it to the preceding layer
    """
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, learning_rate=1.0):
        
        grad_name = "FlipGradient%d" % self.num_calls
        
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * learning_rate]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x) # copy for assign op
            
        self.num_calls += 1
        return y

def random_mini_batches(X, Y, mini_batch_size=200, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data
    Y -- true "label" vector
    mini_batch_size -- size of the mini-batches
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)
    m = X.shape[0]                  # number of training examples
    mini_batches = []
        
    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, 10))

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        end = m - mini_batch_size * int(m / mini_batch_size)
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def visualize_tsne(x, y):
    """
    Display a 2D projection for visualization from the original dataset
    """
    n = x.shape[0]

    model = TSNE(learning_rate=100, n_components=2, random_state=0, n_iter=1000)
    transformed = model.fit_transform(x)
    
    plt.rc('font',family='DejaVu Sans', size=14)
    
    for v, label in zip(range(2), ['source', 'target']):
        idx = [i for i in range(n) if y[i] == v]
        plt.scatter(transformed[idx, 0], transformed[idx, 1], label=label)

    plt.legend()
    plt.show()

