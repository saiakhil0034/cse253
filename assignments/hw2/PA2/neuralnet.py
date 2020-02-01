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

import os, math, gzip, yaml
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
    """
    # img is (60,000 x 784) matrix where rows = samples, cols = pixels in img
    # normalize each column to 0 mean with unit standard deviation (z-score)
    img = (img - np.mean(img,axis=0)) / np.std(img,axis=0)
    return img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    Here we are assuming labels are already from 0 to num_classes,
    if not we need to encode them to 0 to num_classes
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
        # instantiate input
        self.x = a
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
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
        # print(x)
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
        return np.maximum(x,0)


    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        # gradient of sigmoid is s(x)(1-s(x))
        ### think this is wrong, probably needs to be in terms of t and y like last project
        #return self.sigmoid(self.x) * (1 - self.sigmoid(self.x)) 
        return np.multiply(self.y, (1 - self.y))


    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        # gradient of tanh is 4 / (e^x + e^-x)^2
        # return (4 / ((np.exp(self.x) + np.exp(-self.x))**2))
        return 1 - (self.y**2)


    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        # x > 0, gradient = 1
        # x <= 0, gradient = 0
        # output = (self.x > 0)
        # return output.astype(int)
        return 1 * (self.y > 0)



class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units, config):
        """
        Define the architecture and create placeholder.
        """
        # define batch size
        self.batch_size = config['batch_size']
        
        # define momentum per layer for weights and bias
        self.vt_w = 0
        self.vtm1_w = 0
        self.vt_b = 0
        self.vtm1_b = 0
        
        # Declare the Weight matrix
        # weights should be initialized with 0 mean into node j and 
        # std = sqrt(1/m) where m is fan in to next layer
        # axis=0 since we need 0 mean/std into a single node j
        self.w = np.random.normal(0,1,(in_units, out_units))
        self.w = (self.w - np.mean(self.w,axis=0)) / (np.std(self.w,axis=0) * np.sqrt(in_units))
        
        # checker.py
        # initialize weights to standard normal
        # np.random.seed(42)
        # self.w = np.random.normal(0,1,(in_units, out_units))
        
        # Initialize bias to 0, piazza 197
        self.b = np.zeros((1, out_units))
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
        # for each layer, forward pass is just dot(w_j,x_i) + b_j
        # this can generalize through matrix multiplication
        # w shape assumption (in_units,out_units)
        # x shape assumption (matrix where each row is a sample)
        self.a = np.matmul(x,self.w) + self.b 
        return self.a 
        """
        self.x = x
        self.a = (self.x).dot(self.w) + self.b
        return self.a
        """

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        self.d_x = delta.dot(self.w.T) / self.batch_size
        self.d_w = self.x.T.dot(delta) / self.batch_size
        self.d_b = delta.sum(axis=0) / self.batch_size

        return self.d_x

    def __repr__(self):
        # layer num : {self.layer_num},
        return f"output :{self.a[:1]},input : {self.x[:1]})"


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
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1],config))
            if i < len(config['layer_specs']) - 2:
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
        # For each layer call layer.forward(x) where x is the output from
        # previous layer
        self.x = x
        self.targets = targets
        self.y = x
        for l in self.layers:
            self.y = l.forward(self.y)
        
        # perform softmax at end of network
        self.y = softmax(self.y)
            
        return self.y, self.loss(self.y, targets) if targets is not None else self.y         


    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(axis=1)


    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        # take log of y
        ln_y = np.log(logits)
        # multiply by targets and sum along classes axis
        t_ln_y = np.sum(targets * ln_y,axis=1)
        # Compute cost and normalize by number of samples/classes in data
        return (-(np.sum(t_ln_y)) / (targets.shape[0])) #*targets.shape[1]))
        #loss = -(np.multiply(np.log(logits), targets)).sum(axis=1).sum()
        #return loss


    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        delta = self.targets - self.y
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)


def train_and_test(x_train, y_train, test_data, test_target, config, k=10):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    # define congif parameters
    batch_size = config['batch_size']
    epochs = config['epochs']
    alpha = config['learning_rate']
    mom_gamma = config['momentum_gamma']    
    
    ### Step 1: Split images in K subsections
    data, targets = split_training_data(x_train, y_train, k)
    
    # Store loss/accuracy for each epoch per fold so we can calculate average
    # accuracy/loss matrices: shape (k x num_epochs)
    # Plot mean/std of each col    
    train_loss = np.zeros((k,epochs)) 
    train_acc = np.zeros((k,epochs)) 
    val_loss = np.zeros((k,epochs)) 
    val_acc = np.zeros((k,epochs)) 
    test_acc_per_fold = np.zeros(k) 
    
    ### Step 2: Define train and validation data, normalize all data    
    for fold in range(k):
        # create new model for each fold
        model = Neuralnetwork(config)
        # Grab validation set (20 % of data)
        test_idx = (fold + 1) % k
        val_data = np.concatenate((data[fold],data[test_idx]),axis=0)
        val_target = np.concatenate((targets[fold],targets[test_idx]),axis=0)
        # Grab training set
        first_flag = True
        for i in range(k):
            # if key isnt val or test model add to training data
            if (i != fold) and (i != test_idx):
                # if not first training subset 
                if (first_flag):
                    first_flag = False
                    train_data = data[i]
                    train_target = targets[i]
                else:
                    train_data = np.concatenate((train_data,data[i]),axis=0)
                    train_target = np.concatenate((train_target,targets[i]),axis=0)
                           
        # Get training mean and std
        train_mean = np.mean(train_data,axis=0)
        train_std = np.std(train_data,axis=0)
                                  
        # z score each pixel channel
        # value = (old_val - channel mean) / (channel std)
        train_data = (train_data - train_mean) / train_std
        val_data = (val_data - train_mean) / train_std
        test_data = (test_data - train_mean) / train_std         

        ### Step 3: Train Neural Net
        # compute initial training/val/test loss and accuracy
        pred_train, loss_train = model(train_data, train_target)
        train_acc[fold,0] = eval_metrics(pred_train, train_target)
        train_loss[fold,0] = loss_train
        # validation loss/accuracy with updated weights
        pred_val, loss_val = model(val_data, val_target)
        val_acc[fold,0] = eval_metrics(pred_val, val_target)
        val_loss[fold,0] = loss_val        

        # define best model loss per fold
        best_model_loss = math.inf 
        
        """
        # get numbe of weight matrices in model
        num_layers_with_weight_mat = 0
        for layer in model.layers:
            if (hasattr(layer,'activation_type')):
                continue
            else: 
                num_layers_with_weight_mat += 1
        best_weights = [None] * num_layers_with_weight_mat
        best_bias = [None] * num_layers_with_weight_mat
        """
            
        # Call gradient descent to train weights for num_epochs times
        # SDG with momentum and regularization
        for m in range(epochs):
            for i, (tdata, labels) in enumerate(get_data_batch(train_data, train_target, batch_size, shuffle=True)):
                # train model on batch
                pred, loss = model(tdata, labels)
                model.backward()
                # update weights
                for layer in model.layers[::-1]:
                    if (hasattr(layer,'activation_type')):
                        continue
                    else:
                        # Weights
                        # define momentum
                        layer.vt_w = mom_gamma*layer.vtm1_w + layer.d_w
                        # update weights
                        layer.w = layer.w + alpha*layer.vt_w 
                        # update momentum
                        layer.vtm1_w = layer.vt_w
                        # Bias
                        # define momentum
                        layer.vt_b = mom_gamma*layer.vtm1_b + layer.d_b
                        # update bias
                        layer.b = layer.b + alpha*layer.vt_b
                        # update momentum
                        layer.vtm1_b = layer.vt_b
    
            # train loss/accuracy with updated weights
            pred_train, loss_train = model(train_data, train_target)
            train_acc[fold,m] = eval_metrics(pred_train, train_target)
            train_loss[fold,m] = loss_train
            # validation loss/accuracy with updated weights
            pred_val, loss_val = model(val_data, val_target)
            val_acc[fold,m] = eval_metrics(pred_val, val_target)
            val_loss[fold,m] = loss_val
            print(f"Epoch:{m+1}, Accuracy:{val_acc[fold,m+1]}, loss:{val_loss[fold,m+1]}")
    
            # Save model with lowest validation loss and use this to compute best 
            # test performance for this fold
            if (val_loss[fold,m+1] < best_model_loss):
                best_model_loss = val_loss[fold,m+1]
                """
                # save current weights in case best weights
                for layer in model.layers:
                    if (hasattr(layer,'activation_type')):
                        continue
                    else:
                        best_weights[layer_num] = layer.w
                        best_bias[layer_num] = layer.b
                """
                pred_test, loss_test = model(test_data,test_target) 
                test_acc_per_fold[fold] = eval_metrics(pred_test, test_target)

                
    ### Step 4: Plots and Calculations
    avg_test_acc_final = np.mean(test_acc_per_fold)
    avg_test_acc_std_final = np.std(test_acc_per_fold)
    print('Test Accuracy Using Best Model: ' + str(avg_test_acc_final))
    print('Test Accuracy Standard Deviation Using Best Model: ' + str(avg_test_acc_std_final))
    plot_curves(train_loss, val_loss, train_acc, val_acc)



def eval_metrics(y_pred, y_actual):
    """
    evaluation metrics, for now adding only accuracy
    """
    accuracy = np.mean(y_actual.argmax(axis=1) == y_pred.argmax(axis=1)) * 100
    return accuracy


def split_training_data(x_train,y_train,k):
    # split data into k-subsets and store in dictionaries where the key
    # is the subset number and values are the images or targets
    img_data = {}
    target_data = {}
    pixels_per_subset = int(np.floor(x_train.shape[0] / k))
    curr_subset = 0
    for i in range(k):
        # find beginning and ending index for each subset
        begin_idx = int(curr_subset * pixels_per_subset)
        end_idx = int(begin_idx + pixels_per_subset)
        # populate dictionaries
        img_data[curr_subset] = x_train[begin_idx:end_idx,:]
        target_data[curr_subset] = y_train[begin_idx:end_idx,:]
        # update subset and stopping condition
        curr_subset += 1
        if (curr_subset == k):
            break    
    # 
    return img_data, target_data
    
"""
Loss and Performance Plots Function
    General Info: This function plots all needed graphs
    Input: 
        - training_set: 
"""
def plot_curves(train_loss, val_loss, train_acc, val_acc):
    # define x-axis using number of epochs
    x_axis = np.arange(train_loss.shape[1])
    
    # Average the train/val loss and accuracy from all k trials
    # Plot average and std of training and validation curves
    avg_train_loss = np.mean(train_loss,axis=0)
    std_train_loss = np.std(train_loss,axis=0)
    avg_val_loss = np.mean(val_loss,axis=0)
    std_val_loss = np.std(val_loss,axis=0)
    avg_train_acc = np.mean(train_acc,axis=0)
    std_train_acc = np.std(train_acc,axis=0)
    avg_val_acc = np.mean(val_acc,axis=0)
    std_val_acc = np.std(val_acc,axis=0)
    
    # Plot average train/val loss
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x_axis,avg_train_loss,yerr=std_train_loss,errorevery=10)
    ax.errorbar(x_axis,avg_val_loss,yerr=std_val_loss,errorevery=10)
    ax.legend(['Avg Training Loss','Avg Val Loss'])
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Cross Entropy Error')
    ax.set_title('Average Training and Validation Loss')
    ax.grid()
    plt.show()
    
    # plot average train/val accuracy
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x_axis,avg_train_acc,yerr=std_train_acc,errorevery=10)
    ax.errorbar(x_axis,avg_val_acc,yerr=std_val_acc,errorevery=10)
    ax.legend(['Avg Training Accuracy','Avg Val Accuracy'])
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Average Training and Validation Accuracy')
    ax.grid()
    plt.show()


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./config.yaml")

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test = load_data(path="./", mode="t10k")
    num_train_patterns = x_train.shape[0]
    num_test_patterns = x_test.shape[0]

    print(f"number of patterns in training data :{num_train_patterns}")
    print(f"number of patterns in testing data :{num_test_patterns}")

    # train and test the model
    train_and_test(x_train, y_train, x_test,
          y_test, config)
