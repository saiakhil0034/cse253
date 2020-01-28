################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
	# img is (60,000 x 784) matrix where rows = samples, cols = pixels in img
	# normalize each column to 0 mean with unit standard deviation (z-score)
    img = (img - np.mean(img,axis=0)) / np.std(img,axis=0)
    return img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    # labels = {0,1,...,9}
    # labels = (60,000) column vector
    one_hot_labels = np.zeros((labels.shape[0],num_classes))
    for i in range(labels.shape[0]):
        one_hot_labels[i,labels[i]] = 1
    return one_hot_labels


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    # define constant to account for overflow
    c = np.max(x)
    # subtract c from each x to account for overflow
    x = x - c
    # take exponential 
    exp_x = np.exp(x)
    # normalize
    y = exp_x.T / np.sum(exp_x,axis=1)
    # return
    return y.T

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        # instantiate input
        self.x = a
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        return (1 / (1 + np.exp(-x)))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return np.maximum(x,0)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        # gradient of sigmoid is s(x)(1-s(x))
        ### think this is wrong, probably needs to be in terms of t and y like last project
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x)) 

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        # gradient of tanh is 4 / (e^x + e^-x)^2
        return (4 / ((np.exp(self.x) + np.exp(-self.x))**2))

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        # x > 0, gradient = 1
        # x <= 0, gradient = 0
        output = (self.x > 0)
        return output.astype(int)


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        # Declare the Weight matrix 
        # weight matrix shape (rows = num input nodes, cols = num output nodes)
        self.w = np.zeros((in_units,out_units))
        # Create a placeholder for Bias
        # bias vector shape (1, num output nodes)
        self.b = np.zeros((1,out_units)) 
        # Save the input to forward in this
        self.x = None    
        # Save the output of forward pass in this (without activation)
        # a vector shape (1, num output nodes)
        self.a = None    

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        # save input to forward
        self.x = x
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        # for each layer, forward pass is just dot(w_j,x_i) + b_j
        # this can generalize through matrix multiplication
        # w shape assumption (in_units,out_units)
        # x shape assumption (matrix where each row is a sample)
        self.a = np.matmul(x,self.w) + self.b 
        return self.a 

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        raise NotImplementedError("Backprop for Layer not implemented.")


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        # save input to forward
        self.x = x
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        # For each layer call layer.forward(x) where x is the output from
        # previous layer
        for l in self.layers:
            x = l.forward(x)
        self.y = x
        return self.y 

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        # take log of y
        ln_y = np.log(logits)
        # multiply by targets and sum along classes axis
        t_ln_y = np.sum(targets * ln_y,axis=1)
        # Compute cost and normalize by number of samples/classes in data
        return (-(np.sum(t_ln_y)) / (targets.shape[0]*targets.shape[1]))

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        raise NotImplementedError("Backprop not implemented for NeuralNetwork")


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    # Implement Batch Stochastic Gradient Descent (Batch = 128 samples)
    if (x_train.shape[0] % 128 != 0):
        num_batches = np.floor(x_train.shape[0] / 128)
        even_split_size = num_batches * 128
        x_train_batch = np.split(x_train[:even_split_size,:],num_batches,axis=0)
        x_train_batch.append(x_train[-(x_train.shape[0]-even_split_size):,:])
    else:
        x_train_batch = np.split(x_train,x_train.shape[0]/128,axis=0)
    

    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    # run test data through model
    #model_after_forward_prop = model.__call__(X_test)
    # get model output
    #model_output = model_after_forward_prop.y
    # calculate test error
    
    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    # setting 20% of data as validation set
    num_valid_samples = int(np.floor(x_train.shape[0]*(.2)))
    end_idx = x_train.shape[0]
    x_valid = x_train[-num_valid_samples:,:]
    y_valid = y_train[-num_valid_samples:,:]
    x_train = x_train[:-num_valid_samples,:]
    y_train = y_train[:-num_valid_samples,:]

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
