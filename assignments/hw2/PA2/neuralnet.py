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
    # img is (60,000 x 784) matrix where rows = samples, cols = pixels in img

    img_cp = img.copy()

    # first min max normalisation to remove brightness effects
    min_val = img_cp.min(axis=1).reshape(-1, 1)
    max_val = img_cp.max(axis=1).reshape(-1, 1)
    minmax_norm_img = (img_cp - min_val) / (max_val - min_val)

    # then centerring the datset :
    # normalize each column to 0 mean with unit standard deviation (z-score)
    norm_img = (minmax_norm_img - np.mean(minmax_norm_img, axis=0)
                ) / np.std(minmax_norm_img, axis=0)
    return norm_img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    Here we are assuming labels are already from 0 to num_classes,
    if not we need to encode them to 0 to num_classes
    """
    # labels = {0,1,...,9}
    # labels = (60,000) column vector

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


def split_training_data(x_train, y_train, k):
    # split data into k-subsets and store in dictionaries where the key
    # is the subset number and values are the images or targets
    img_data = {}
    target_data = {}
    images_per_subset = int(np.floor(x_train.shape[0] / k))
    curr_subset = 0
    for i in range(k):
        # find beginning and ending index for each subset
        begin_idx = int(curr_subset * images_per_subset)
        end_idx = int(begin_idx + images_per_subset)
        # populate dictionaries
        img_data[curr_subset] = x_train[begin_idx:end_idx, :]
        target_data[curr_subset] = y_train[begin_idx:end_idx, :]
        # update subset and stopping condition
        curr_subset += 1
        if (curr_subset == k):
            break
    return img_data, target_data


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
        """
        This method allows your instances to be callable.
        """
        # instantiate input
        self.x = a
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
        # x > 0, gradient = 1
        # x <= 0, gradient = 0
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
        self.batch_size = config["batch_size"]  # define batch size

        # Declare the Weight matrix
        # weight matrix shape (rows = num input nodes, cols = num output nodes)
        # weights should be initialized with 0 mean into node j and
        # std = sqrt(1/m) where m is fan in to next layer
        # axis=0 since we need 0 mean/std into a single node j
        self.w = np.random.randn(in_units, out_units)
        self.w = (self.w - np.mean(self.w, axis=0)) / \
            (np.std(self.w, axis=0) * np.sqrt(in_units))
        # bias vector shape (1, num output nodes)
        # Initialize bias to 0, piazza 197
        self.b = np.zeros((1, out_units))

        self.x = None       # input to forward
        # a vector shape (1, num output nodes)
        self.a = None       # output of forward pass(without activation)

        self.d_x = None     # gradient w.r.t x in this
        self.d_w = None     # gradient w.r.t w in this
        self.d_b = None     # gradient w.r.t b in this
        self.vw = 0       # Momentum term
        self.vb = 0       # Momentum term

    def __call__(self, x):
        """Making layer callable."""
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """

        self.x = x
        # for each layer, forward pass is just dot(w_j,x_i) + b_j
        # this can generalize through matrix multiplication
        # w shape assumption (in_units,out_units)
        # x shape assumption (matrix where each row is a sample)
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
        """Making NeuralNetwork callable."""
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """

        self.x = x
        self.targets = targets

        # For each layer call layer.forward(x) where x is the output from
        # previous layer

        a = x
        for layer in self.layers:
            a = layer.forward(a)

        # perform softmax at end of network
        self.y = softmax(a)

        return self.y, self.loss(self.y, targets) if (targets is not None) else self.y

    def predict(self, x):
        logits, _ = self.forward(x)
        return logits.argmax(axis=1)

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        # take log of y
        ln_y = np.log(logits)
        # multiply by targets and sum along classes axis
        #
        t_ln_y = np.sum(np.multiply(targets, ln_y), axis=1)
        # Compute cost and normalize by number of samples in data
        return t_ln_y.mean()

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        # to avg the gradient per loss of a smaple in future calculations
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


def eval_metrics(y_pred, y_actual):
    """
    evaluation metrics, for now adding only accuracy
    """
    accuracy = np.mean(y_actual.argmax(axis=1) == y_pred.argmax(axis=1)) * 100
    return accuracy

# class Model(object):
#     """docstring for Model"""
#     def __init__(self, arg):
#         super(Model, self).__init__()
#         self.arg = arg

#     def validation_sets(self):


#     def test(self):


#     def plotter(self):

def cv_fold_data(data, targets, fold, k):
    # Grab validation set (20 % of data)
    # test_idx = (fold + 1) % k
    val_data = data[fold]  # np.concatenate((, data[test_idx]), axis=0)
    val_target = targets[fold]  # np.concatenate((, targets[test_idx]), axis=0)

    # Grab training set
    train_data = np.vstack([data[i] for i in range(k) if (i != fold)])
    train_target = np.vstack([targets[i] for i in range(k) if (i != fold)])

    return train_data, train_target, val_data, val_target


def train_and_test(x_train, y_train, test_data, test_target, config, k=10):
    """
    Train a nd text your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    # define congif parameters
    epochs = config['epochs']

    # Step 1: Split images in K subsections
    input_data, targets = split_training_data(x_train, y_train, k)

    # Store loss/accuracy for each epoch per fold so we can calculate average
    # accuracy/loss matrices: shape (k x num_epochs)
    # Plot mean/std of each col
    train_loss = np.zeros((k, epochs))
    train_acc = np.zeros((k, epochs))
    val_loss = np.zeros((k, epochs))
    val_acc = np.zeros((k, epochs))

    test_acc_per_fold = np.zeros(k)

    model = Neuralnetwork(config)
    print(f"\nCurrent Model : {model}")
    print(f"used config : \n{config}\n")

    # Step 2: Define train and validation data, normalize all data
    for fold in range(k):
        print(f"working with fold : {fold}")
        # create new model for each fold
        model = Neuralnetwork(config)

        train_data, train_target, val_data, val_target = cv_fold_data(
            input_data, targets, fold, k)
        print(train_data.shape)

        # z-scoring the fold dataset (optional)
        # Get training mean and std
        # train_mean = np.mean(train_data, axis=0)
        # train_std = np.std(train_data, axis=0)

        # # z score each pixel channel
        # # value = (old_val - channel mean) / (channel std)
        # train_data = (train_data - train_mean) / train_std
        # val_data = (val_data - train_mean) / train_std
        # test_data = (test_data - train_mean) / train_std

        # Step 3: Train Neural Net
        # compute initial training/val/test loss and accuracy
        pred_train, loss_train = model(train_data, train_target)
        train_acc[fold, 0] = eval_metrics(pred_train, train_target)
        train_loss[fold, 0] = loss_train
        # validation loss/accuracy with updated weights
        pred_val, loss_val = model(val_data, val_target)
        val_acc[fold, 0] = eval_metrics(pred_val, val_target)
        val_loss[fold, 0] = loss_val

        # define best model loss per fold
        best_model_loss = float("inf")

        # Call gradient descent to train weights for num_epochs times
        # Minibatch GD with momentum and regularization

        for epoch in range(epochs):
            for i, (data, labels) in enumerate(get_data_batch(train_data, train_target, config["batch_size"], shuffle=True)):
                # print(data.shape)
                pred, loss = model(data, labels)  # forward pass on batch
                model.backward()  # backward pass
                model.update_network()  # update weights

            # train loss/accuracy with updated weights
            pred_train, loss_train = model(train_data, train_target)
            train_acc[fold, epoch] = eval_metrics(pred_train, train_target)
            train_loss[fold, epoch] = loss_train

            # validation loss/accuracy with updated weights
            pred_val, loss_val = model(val_data, val_target)
            val_acc[fold, epoch] = eval_metrics(pred_val, val_target)
            val_loss[fold, epoch] = loss_val

            # early stopping criterion
            # limit on epoch to avoid  stopping for initial random jumps
            if config["early_stop"]:
                if (epoch > config["early_stop_epoch"]):
                    # loss is in order of 1, hence to avoid nose this 0.1
                    if (loss_val > val_loss[fold, epoch - 1] + 0.1):
                        break

            # print(
            #     f"Epoch:{epoch}, Accuracy:{val_acc[fold,epoch]}, loss:{val_loss[fold,epoch]}")

            # Save model with lowest validation loss and use this to compute best
            # test performance for this fold
            if (val_loss[fold, epoch] < best_model_loss):
                best_model_loss = val_loss[fold, epoch]
                """
                # save current weights in case best weights
                for layer in model.layers:
                    if (hasattr(layer,'activation_type')):
                        continue
                    else:
                        best_weights[layer_num] = layer.w
                        best_bias[layer_num] = layer.b
                """
                pred_test, loss_test = model(test_data, test_target)
                test_acc_per_fold[fold] = eval_metrics(pred_test, test_target)

    # Calculating mean and stds across folds
    avg_train_loss = np.mean(train_loss, axis=0)
    std_train_loss = np.std(train_loss, axis=0)
    avg_val_loss = np.mean(val_loss, axis=0)
    std_val_loss = np.std(val_loss, axis=0)
    avg_train_acc = np.mean(train_acc, axis=0)
    std_train_acc = np.std(train_acc, axis=0)
    avg_val_acc = np.mean(val_acc, axis=0)
    std_val_acc = np.std(val_acc, axis=0)

    # Step 4: Plots and Calculations
    avg_test_acc_final = np.mean(test_acc_per_fold)
    avg_test_acc_std_final = np.std(test_acc_per_fold)
    print('Test Accuracy Using Best Model: ' + str(avg_test_acc_final))
    print('Test Accuracy Standard Deviation Using Best Model: ' +
          str(avg_test_acc_std_final))

    # Plot average and std of training and validation curves
    plot_curves(avg_train_loss, std_train_loss, avg_val_loss,  std_val_loss,
                avg_train_acc, std_train_acc, avg_val_acc,  std_val_acc)


def test(model, X_test, y_test):
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
    for data, error in zip(y_data_arr, e_data_arr):
        ax.errorbar(x_data, data, yerr=error, errorevery=10)

    ax.legend(kwargs["legend"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])
    ax.grid()
    plt.show()


def plot_curves(avg_train_loss, std_train_loss, avg_val_loss,  std_val_loss,
                avg_train_acc, std_train_acc, avg_val_acc,  std_val_acc):
    """
    Loss and Performance Plots Function
    General Info: This function plots all needed graphs
    Input:
        - training_set: numpy array of floats
    """
    # define x-axis using number of epochs
    x_axis = np.arange(avg_train_loss.shape[0])

    # Plot average train/val loss
    loss_data_arr = [avg_train_loss, avg_val_loss]
    loss_kwargs = {
        "e_data_arr": [std_train_loss, std_val_loss],
        "legend": ['Avg Training Loss', 'Avg Val Loss'],
        "x_label": 'Number of Epochs',
        "y_label": 'Avg Negative Cross Entropy Error',
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


def check_num_grad(x_train, y_train, config):
    # Choosing one hidden layer for these calculations
    x_data = []
    y_data = []
    for i in range(y_train.shape[1]):
        x_data.append(x_train[(y_train[:, i] == 1)][:10])
        y_data.append(y_train[(y_train[:, i] == 1)][:10])

    x_data = np.vstack(x_data)
    y_data = np.vstack(y_data)

    # Create the model
    model = Neuralnetwork(config)

    chosen_nodes = [Node(2, (0, 5), "b"), Node(0, (0, 50), "b"),
                    Node(2, (30, 7), "w"), Node(2, (70, 3), "w"),
                    Node(0, (300, 30), "w"), Node(0, (500, 70), "w")]

    for node in chosen_nodes:
        print(f"layer : {node.layer}, position : {node.ix}")
        # print(getattr(model.layers[node.layer], node.type))
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
        print(f"differece : {num_grad - bp_grad}\n")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

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
    # check_num_grad(x_train, y_train, config)

    # # Create splits for validation data here.
    # x_train, y_train, x_valid, y_valid = validation_split(
    #     x_train, y_train, split=0.2, cross_validation=False)

    # # print(x_train[:1], x_train[:1].min(), x_train[:1].max())

    # # train the model
    # train(model, x_train, y_train, x_valid,
    #       y_valid, config, epochs=100)

    # test_acc = test(model, x_test, y_test)
    # print(test_acc)

    # train and test the model
    train_and_test(x_train, y_train, x_test,
                   y_test, config)
