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

import os
import gzip
import yaml
import numpy as np
import matplotlib.pyplot as plt


class EvalMetrics(object):
    """docstring for EvalMetrics"""

    def __init__(self, arg):
        super(EvalMetrics, self).__init__()
        self.loss = arg[0]
        self.accuracy = arg[1]


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    Here we are doing min max normalisation
    """
    img_cp = img.copy()
    min_val = img_cp.min(axis=1).reshape(-1, 1)
    max_val = img_cp.max(axis=1).reshape(-1, 1)
    norm_img = (img_cp - min_val) / (max_val - min_val)
    return norm_img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    Here we are assuming labels are already from 0 to num_classes,
    if not we need to encode them to 0 to num_classes
    """
    num_patterns = len(labels)
    ohe_arr = np.zeros((num_patterns, num_classes))
    ohe_arr[np.arange(num_patterns), labels] = 1
    return ohe_arr


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
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def reverse_one_hot_encoding(labels):
    """
    reversing the one hot encoding
    """
    return labels.argmax(axis=1)


def plot_labels(data, labels):
    """
    visulaising the labels
    """
    pure_labels = reverse_one_hot_encoding(labels)
    plot_data = [data[list(pure_labels).index(x)]
                 for x in sorted(set(pure_labels))]

    rows = 2
    columns = 5
    fig, ax = plt.subplots(rows, columns, figsize=(10, 6))

    count = 0
    for i in range(rows):
        for j in range(columns):
            ax[i][j].imshow(plot_data[count].reshape(28, 28))
            ax[i][j].set_title(f"label : {count}")
            count += 1
    fig.suptitle("Plotting imgs for different labels")
    fig.tight_layout()
    plt.show()


def validation_split(train_x, train_y, split=0.2, cross_validation=False):
    """
    splitting the training data for validation (cross validation)
    """
    num_train_patterns = train_x.shape[0]
    if cross_validation:
        pass
    else:
        indices = np.random.permutation(num_train_patterns)
        num_valid = int(split * num_train_patterns)
        training_idx, valid_idx = indices[:-num_valid], indices[-num_valid:]
        x_train, x_valid = train_x[training_idx, :], train_x[valid_idx, :]
        y_train, y_valid = train_y[training_idx, :], train_y[valid_idx, :]
        return x_train, y_train, x_valid, y_valid


def get_data_batch(data, targets, batchsize, shuffle=True):
    """
    # Generator for batches of data
    """
    n = data.shape[0]
    if shuffle:
        indices = np.random.permutation(n)
    else:
        indices = np.arange(n)
    for i in range(int(np.ceil(n / batchsize))):
        ixs = indices[i * batchsize: min(n, (i + 1) * batchsize)]
        yield data[ixs], targets[ixs]


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    k = np.max(x, axis=1).reshape(-1, 1)  # To avoid overflow
    x = x - k
    numerator = np.exp(x)
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)
    #print(x)

    return np.divide(numerator, denominator)


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input,outputs. These will be used for computing gradients.
        self.x = None
        self.y = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        # print(a)
        self.x = a
        self.y = None
        if self.activation_type == "sigmoid":
            self.y = self.sigmoid(a)

        elif self.activation_type == "tanh":
            self.y = self.tanh(a)

        elif self.activation_type == "ReLU":
            self.y = self.ReLU(a)

        return self.y

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
        #print(x)
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Implement tanh here.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        return x * (x > 0)

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return np.multiply(self.y, (1 - self.y))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - (self.y**2)

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return 1 * (self.y > 0)


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units, layer_num):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.layer_num = layer_num
        self.num_ins = in_units
        self.num_ons = out_units
        # Declare the Weight matrix
        self.w = np.random.rand(in_units, out_units)
        # Create a placeholder for Bias
        self.b = np.random.rand(1, out_units)
        self.x = None    # Save the input to forward in this
        # Save the output of forward pass in this (without activation)
        self.a = None

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        self.x = x
        self.a = (self.x).dot(self.w) + self.b
        return self.a

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.dx = delta.dot(self.w.T)
        self.dw = self.x.T.dot(delta)
        self.db = delta.mean(axis=0)

        self.w -= self.dw
        self.b -= self.db
        return self.dx


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
        self.num_w_layers = len(config['layer_specs']) - 1

        # Add layers specified by layer_specs.
        for i in range(self.num_w_layers):
            self.layers.append(
                Layer(config['layer_specs'][i], config['layer_specs'][i + 1], i))
            if i < (self.num_w_layers - 1):
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.x = x
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        self.y = softmax(a)

        self.targets = targets
        return self.y, self.loss(self.y, targets) if targets is not None else self.y

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(axis=1)

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        loss = (-np.multiply(np.log(logits), targets)).sum(axis=1).mean()
        return loss

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        delta = self.targets - self.y
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)


def train(model, x_train, y_train, x_valid, y_valid, config, epochs=10, batch_size=50):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    eval_arr = []
    for epoch in range(epochs):

        for i, (data, labels) in enumerate(get_data_batch(x_train, y_train, batch_size, shuffle=True)):
            pred, loss = model(data, labels)
            model.backward()

        pred_valid, loss_valid = model(x_valid, y_valid)
        acc_valid = eval_metrics(pred_valid, y_valid)
        print(
            f"Epoch:{epoch+1}, Accuracy:{acc_valid}, -logloss:{loss_valid}")
        # early stopping criterion
        # limit on epoch to avoid  stopping forinitial random jumps
        if (epoch > 10):
            if (acc_valid < eval_arr[-1].accuracy):
                break

        eval_arr.append(EvalMetrics([loss_valid, acc_valid]))


def eval_metrics(y_pred, y_actual):
    """
    evaluation metrics, for now adding only accuracy
    """
    accuracy = np.mean(y_actual.argmax(axis=1) == y_pred.argmax(axis=1)) * 100
    return accuracy


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """
    batchsize = 50
    correct = 0.
    for data, label in DataBatch(X_test, y_test, batchsize, shuffle=False):
        prediction = model.predict(data)
        correct += np.sum(prediction == label.argmax(axis=1))
    return correct * 100 / testData.shape[0]


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Create the model
    model = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test = load_data(path="./", mode="t10k")
    num_train_patterns = x_train.shape[0]
    num_test_patterns = x_test.shape[0]

    print(f"number of patterns in training data :{num_train_patterns}")
    print(f"number of patterns in testing data :{num_test_patterns}")

    # Visualising the labels
    # plot_labels(x_train, y_train)

    # Create splits for validation data here.
    x_train, y_train, x_valid, y_valid = validation_split(
        x_train, y_train, split=0.2, cross_validation=False)

    # train the model
    train(model, x_train, y_train, x_valid,
          y_valid, config, epochs=10, batch_size=50)

    # test_acc = test(model, x_test, y_test)
