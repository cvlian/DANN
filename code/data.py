import os
import pickle
from utils import *

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data

class Dataset:

    def getTask(self):
        return self.task

    def showSamples(self, nrows, ncols):
        """
        Plot nrows x ncols images
        """
        fig, axes = plt.subplots(nrows, ncols)
        for i, ax in enumerate(axes.flat): 
            ax.imshow(self.x[i,:])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(np.argmax(self.y[i]))
        
        plt.show()

        
class MNISTdata(Dataset):
    """
    MNIST dataset
    
    A large collection of monochrome images of handwritten digits
    
    It has a training set of 55,000 examples, and a test set of 10,000 examples
    """

    def __init__(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        x_train = np.reshape(mnist.train.images, [-1, 28, 28, 1])
        x_val = np.reshape(mnist.test.images, [-1, 28, 28, 1])
        x_train = np.concatenate([x_train, x_train, x_train], 3)
        x_val = np.concatenate([x_val, x_val, x_val], 3)
        
        print("MNIST : Training Set", x_train.shape)
        print("MNIST : Test Set", x_val.shape)
        
        # Calculate the total number of images
        num_images = x_train.shape[0] + x_val.shape[0]
        print("MNIST : Total Number of Images", num_images)
        
        self.task = {'name':'mnist', 'x_train':x_train, 'x_val':x_val, 'y_train':mnist.train.labels, 'y_val':mnist.test.labels}


class MNIST_Mdata(Dataset):
    """
    MNIST-M dataset
    
    This dataset is created by combining MNIST digits with the patches
    randomly extracted from color photos of BSDS500 as their background
    
    It contains 55,000 training and 10,000 test images as well
    """
    
    def __init__(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        mnistm = pickle.load(open('../dataset/mnistm_data.pkl', 'rb'), encoding='latin1')

        x_train = np.reshape(mnistm['train'], [-1, 28, 28, 3])/255
        x_val = np.reshape(mnistm['test'], [-1, 28, 28, 3])/255
        
        print("MNIST-M : Training Set", x_train.shape)
        print("MNIST-M : Test Set", x_val.shape)

        # Calculate the total number of images
        num_images = x_train.shape[0] + x_val.shape[0]
        print("MNIST-M : Total Number of Images", num_images)
        
        self.task = {'name':"mnist-m", 'x_train':x_train, 'x_val':x_val, 'y_train':mnist.train.labels, 'y_val':mnist.test.labels}
