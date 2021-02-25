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

    model = TSNE(learning_rate=500, n_components=2, random_state=0, n_iter=250, init='pca')
    transformed = model.fit_transform(x)
    
    plt.rc('font',family='DejaVu Sans', size=14)
    
    for v, label in zip(range(2), ['source', 'target']):
        idx = [i for i in range(n) if y[i] == v]
        plt.scatter(transformed[idx, 0], transformed[idx, 1], label=label)

    plt.legend()
    plt.show()

def visualize_sole_acc(res, maxiter=10, prev_res=None):
    """
    Plot accuracy (one model)
    """
    plt.rc('font',family='DejaVu Sans', size=16)
    fig=plt.figure(figsize=(5, 4.5))
    ax=fig.add_axes([0,0,1,1])
    
    for task, reports in res.items() :
        if prev_res != None and task in prev_res :
            plt.plot(list(range(0, maxiter+1)), [prev_res[task][-1]]+reports[:maxiter], label=task, markersize=8, linewidth=3, clip_on=False)
        else :
            plt.plot(list(range(0, maxiter+1)), [0.0]+reports[:maxiter], label=task, markersize=8, linewidth=3, clip_on=False)

    ax.set_xlim([0, maxiter])
    ax.set_ylim([0.0, 1.0])
    ax.set_xticks(list(range(0, maxiter+1, maxiter//5)))
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(range(0, maxiter+1, maxiter//10), minor=True)
    ax.set_yticks([i/10 for i in range(0, 11)], minor=True)
    ax.set_xlabel("Epochs", fontsize=20)
    ax.set_ylabel("Accuracy", fontsize=20)
    ax.legend(markerscale=1, fontsize=16, loc='lower right')
    ax.grid(which='both', color='#BDBDBD', linestyle='--', linewidth=1)
    plt.rc('font',family='DejaVu Sans', size=16)

    plt.show()

def visualize_multi_acc(models, maxiter=10, tasks=3):
    """
    Plot accuracy (multiple models)
    """
    fig, axs = plt.subplots(tasks, tasks, figsize=(12, 11.5))
    plt.rc('font',family='DejaVu Sans', size=14)
    
    for i in range(tasks) :
        for j in range(0, i):
            axs[i, j].axis('off')
        for j in range(i, tasks):
            for task_name, reports in models[i].res[j-i].items() :
                if j > i and task_name in models[i].res[j-i-1] :
                    axs[i, j].plot(list(range(0, maxiter+1)),
                                   [models[i].res[j-i-1][task_name][-1]]+reports[:maxiter],
                                   label=task_name, linewidth=3, clip_on=False)
                else :
                    axs[i, j].plot(list(range(0, maxiter+1)), 
                                   [0.0]+reports[:maxiter], 
                                   label=task_name, linewidth=3, clip_on=False)
                axs[i, j].set_xlim([0, maxiter])
                axs[i, j].set_ylim([0.0, 1.0])
                axs[i, j].set_xticklabels(list(range(0, maxiter+1, maxiter//5)), fontsize=14)
                axs[i, j].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
                axs[i, j].set_xticks(range(0, maxiter+1, maxiter//10), minor=True)
                axs[i, j].set_yticks([i/10 for i in range(0, 11)], minor=True)
                if i == j :
                    axs[i, j].set_xlabel("Epochs", fontsize=18)
                    axs[i, j].set_ylabel("Accuracy", fontsize=18)
                axs[i, j].legend(markerscale=1, fontsize=14, loc='lower right')
                axs[i, j].grid(which='both', color='#BDBDBD', linestyle='--', linewidth=1)
    
    fig.tight_layout()
    plt.show()