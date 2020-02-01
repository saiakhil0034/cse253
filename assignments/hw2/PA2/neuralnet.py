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
    """Evaluation Metrics"""

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
    """reversing the one hot encoding"""

    return labels.argmax(axis=1)


def plot_labels(data, labels):
    """visulaising the labels"""
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
    """splitting the training data for validation (cross validation)"""
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
    """# Generator for batches of data"""

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
    # print(x)
    numerator = np.exp(x)
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)
    # print(x)
    return np.divide(numerator, denominator)


class Activation(object):
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
        """Initialize activation type and placeholders here."""
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input,outputs. These will be used for computing gradients.
        self.x = None
        self.y = None

    def __call__(self, a):
        """This method allows your instances to be callable."""
        return self.forward(a)

    def forward(self, a):
        """Compute the forward pass."""

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
        """Compute the backward pass."""
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """Implement the sigmoid activation here."""
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """Implement tanh here."""
        return np.tanh(x)

    def ReLU(self, x):
        """Implement ReLU here."""
        return x * (x > 0)

    def grad_sigmoid(self):
        """Compute the gradient for sigmoid here."""
        return np.multiply(self.y, (1 - self.y))

    def grad_tanh(self):
        """Compute the gradient for tanh here."""
        return 1 - (self.y**2)

    def grad_ReLU(self):
        """Compute the gradient for ReLU here."""
        return 1 * (self.y > 0)

    def __repr__(self):
        return f" layer activation : {self.activation_type}"


class Layer(object):
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units, layer_num, config):  # layer_num
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.layer_num = layer_num
        self.num_ins = in_units
        self.num_ons = out_units
        self.lr = config["learning_rate"]
        self.mc = config["momentum_gamma"] if config["momentum"] else 0
        self.rc = config["L2_penalty"]
        self.batch_size = config["batch_size"]

        # Declare the Weight matrix
        self.w = np.random.randn(in_units, out_units)
        self.w = (self.w - np.mean(self.w, axis=0)) / \
            (np.std(self.w, axis=0) * np.sqrt(in_units))
        self.b = np.zeros((1, out_units))

        self.x = None       # input to forward
        self.a = None       # output of forward pass(without activation)

        self.d_x = None     # gradient w.r.t x in this
        self.d_w = None     # gradient w.r.t w in this
        self.d_b = None     # gradient w.r.t b in this
        self.vw = 0       # Momentum term
        self.vb = 0       # Momentum term

    def __call__(self, x):
        """Make layer callable."""
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
        # We want to reduce overfitting by regularisation and bias doesn't have any effect on overfitting
        # Thus, regularisation term is only in weights, not in biases

        self.d_x = delta.dot(self.w.T)  # / self.batch_size
        self.d_w = self.x.T.dot(delta) \
            - self.rc * (self.w)  # / self.batch_size  # regularisation term
        self.d_b = delta.sum(axis=0).reshape(1, -1)  # / self.batch_size

        return self.d_x

    def update_weights(self):
        """
        update weigths using gradient descent
        """

        self.vw = self.mc * self.vw + self.lr * self.d_w
        self.vb = self.mc * self.vb + self.lr * self.d_b

        self.w += self.vw
        self.b += self.vb

    def __repr__(self):
        return f" layer num : {self.layer_num}, shape : {self.num_ins,self.num_ons}"


class Neuralnetwork(object):
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
        self.config = config

        # Add layers specified by layer_specs.
        for i in range(self.num_w_layers):
            self.layers.append(
                Layer(config['layer_specs'][i], config['layer_specs'][i + 1], i, config))
            if i < (self.num_w_layers - 1):
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """Make NeuralNetwork callable."""

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
        return self.y, self.loss(self.y, targets) if (targets is not None) else self.y

    def predict(self, x):
        logits, _ = self.forward(x)
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
        delta = (self.targets - self.y) / self.y.shape[0]
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)

    def update_network(self):
        """
        updatting weights of the network
        """
        for layer in self.layers:
            if layer.__class__.__name__ == "Layer":
                layer.update_weights()

    def __repr__(self):
        hidden_layer_str = "\n\t\t".join([repr(i) for i in self.layers])
        return f"Neural Network \n\
-----------------------------------------------------\n\
Feed forward Neural Network with {self.num_w_layers-1} hidden layers\n\
    number of inputs : {self.layers[0].num_ins}\n\
    number of classes : {self.layers[-1].num_ons}\n\
    hidden layers :\n\t\t{hidden_layer_str}\n\
----------------------------------------------------\n"


def train(model, x_train, y_train, x_valid, y_valid, config, epochs=10):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    print(f"\nCurrent Model : {model}")
    print(f"used config : \n{config}\n")
    eval_arr = []
    for epoch in range(epochs):

        for i, (data, labels) in enumerate(get_data_batch(x_train, y_train, config["batch_size"], shuffle=True)):
            pred, loss = model(data, labels)
            model.backward()
            model.update_network()

        pred_valid, loss_valid = model(x_valid, y_valid)
        acc_valid = eval_metrics(pred_valid, y_valid)
        print(
            f"Epoch:{epoch+1:3}, Accuracy:{acc_valid:7.3f}, -logloss:{loss_valid:.3f}")

        # early stopping criterion
        # limit on epoch to avoid  stopping for initial random jumps
        if config["early_stop"]:
            if (epoch > config["early_stop_epoch"]):
                if (loss_valid > eval_arr[-1].loss + 0.1):
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
    for data, label in get_data_batch(X_test, y_test, batchsize, shuffle=False):
        prediction = model.predict(data)
        correct += np.sum(prediction == label.argmax(axis=1))
    return correct * 100 / y_test.shape[0]


def plot(x_data, y_data_arr, e_data_arr, **kwargs):
    """ Plotting function """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for data, error in zip(y_data_arr, kwargs["e_data_arr"]):
        ax.errorbar(x_axis, data, yerr=error, errorevery=10)

    ax.legend(kwargs["legend"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])
    ax.grid()
    plt.show()


def plot_curves(train_loss, val_loss, train_acc, val_acc):
    """
    Loss and Performance Plots Function
    General Info: This function plots all needed graphs
    Input:
        - training_set: numpy array of floats
    """
    # define x-axis using number of epochs
    x_axis = np.arange(train_loss.shape[1])

    # Average the train/val loss and accuracy from all k trials
    # Plot average and std of training and validation curves
    avg_train_loss = np.mean(train_loss, axis=0)
    std_train_loss = np.std(train_loss, axis=0)
    avg_val_loss = np.mean(val_loss, axis=0)
    std_val_loss = np.std(val_loss, axis=0)
    avg_train_acc = np.mean(train_acc, axis=0)
    std_train_acc = np.std(train_acc, axis=0)
    avg_val_acc = np.mean(val_acc, axis=0)
    std_val_acc = np.std(val_acc, axis=0)

    # Plot average train/val loss
    loss_data_arr = [avg_train_loss, avg_val_loss]
    loss_kwargs = {
        "e_data_arr": [std_train_loss, std_val_loss],
        "legend": ['Avg Training Loss', 'Avg Val Loss'],
        "x_label": 'Number of Epochs',
        "y_label": 'Negative Cross Entropy Error',
        "title": "'Average Training and Validation Loss'"
    }
    plot(x_axis, loss_data_arr, **loss_kwargs)

    # plot average train/val accuracy
    acc_data_arr = [avg_train_acc, avg_val_acc]
    acc_kwargs = {
        "e_data_arr": [std_train_acc, std_val_acc],
        "legend": ['Avg Training Accuracy', 'Avg Val Accuracy'],
        "x_label": 'Number of Epochs',
        "y_label": 'Accuracy',
        "title": "'Average Training and Validation Accuracy'"
    }
    plot(x_axis, acc_data_arr, **acc_kwargs)


class Node(object):
    """Node class for storing layer,position and type used in part b"""

    def __init__(self, layer, ix, type):
        super(Node, self).__init__()
        self.layer = layer
        self.ix = ix
        self.type = type


def check_num_grad(x_train, y_train, model, config):
    # Choosing one hidden layer for these calculations
    x_data = []
    y_data = []
    for i in range(y_train.shape[1]):
        x_data.append(x_train[(y_train[:, i] == 1)][:10])
        y_data.append(y_train[(y_train[:, i] == 1)][:10])

    x_data = np.vstack(x_data)
    y_data = np.vstack(y_data)

    chosen_nodes = [Node(2, (0, 5), "b"), Node(0, (0, 50), "b"),
                    Node(2, (30, 7), "w"), Node(2, (70, 3), "w"),
                    Node(0, (300, 30), "w"), Node(0, (500, 70), "w")]

    for node in chosen_nodes:
        print(f"layer : {node.layer}, position : {node.ix}")
        #print(getattr(model.layers[node.layer], node.type))
        getattr(model.layers[node.layer], node.type)[node.ix] -= 1e-2
        _, loss1 = model(x_data, y_data)
        # since we are subtracted epsilon earlier
        getattr(model.layers[node.layer], node.type)[node.ix] += 2e-2
        _, loss2 = model(x_data, y_data)
        num_grad = (loss2 - loss1) / (2e-2)
        # brining back to original
        getattr(model.layers[node.layer], node.type)[node.ix] -= 1e-2
        model(x_data, y_data)
        model.backward()
        bp_grad = -getattr(model.layers[node.layer], f"d_{node.type}")[node.ix]

        print(f"numerical gradient : {num_grad}")
        print(f"gradient from backprop : {bp_grad}")
        print(f"differece : {num_grad - bp_grad}")


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

    print("Data :")
    print(
        f"number of patterns in training data :{num_train_patterns},testing data :{num_test_patterns}")

    # Visualising the labels
    # plot_labels(x_train, y_train)

    # Check numerical gradient approximation
    check_num_grad(x_train, y_train, model, config)

    # # Create splits for validation data here.
    # x_train, y_train, x_valid, y_valid = validation_split(
    #     x_train, y_train, split=0.2, cross_validation=False)

    # # print(x_train[:1], x_train[:1].min(), x_train[:1].max())

    # # train the model
    # train(model, x_train, y_train, x_valid,
    #       y_valid, config, epochs=100)

    # test_acc = test(model, x_test, y_test)
    # print(test_acc)
