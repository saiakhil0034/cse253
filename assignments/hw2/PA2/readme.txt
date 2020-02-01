# CSE 253 Assignment 2
# Students : Alexander Behnke(A53304866, ), Sai Akhil Suggu (A53284020, ssuggu@ucsd.edu),
# Professor : Prof. Gary Cottrell
# Last change date : Feb 1st, 2020

runnable_file : neuralnet.py
# How to run the file (make sure you have data and config files in same path)
python3 -m neuralnet.py

# for different parts of the question we just need to run the file with different configs

# classes
Neural network :
    This class has architecture of neural network
    Methods available:
        __init__        : constructor
        __call__        : to make class callable
        store           : to store the network weights and config
        makeit          : to load the network with particular weights and config
        forward         : to claculate output for given input
        predict         : to predict the output class
        loss            : claculates the loss given probs and labels
        backward        : backward pass,, calculation of gradients
        update_network  : to update the network weights
        __repr__        : to print the network
Layer :
    This class has the design of linear layer of NN
    Methods available:
        __init__        : constructor
        __call__        : to make class callable
        forward         : to claculate output for given input
        backward        : backward pass, calculation of gradients
        update_weights  : to update the layer weights
        __repr__        : to print the layer

Activation:
    This class has the design of activation layer of NN.
    Activation function availble : sigmoid, ReLu, tanh
    Methods available:
        __init__        : constructor
        __call__        : to make class callable
        forward         : to claculate output for given input
        backward        : backward pass, calculation of gradients

EvalMetrics
    This class is defined to have all evalution metrics.
    Currently, we have only loss and accuracy
    Methods available: None

Node
    This class is written just for part b of programming assigment.
    This has 3 attributes,namely layer number, layer type and position of node in the layer
    Methods available: None

# Functions
load_config                 : returns the config given path
load_data                   : loads the data
normalize_data              : normalises the data
one_hot_encoding            : one hot encodes the label data
reverse_one_hot_encoding    : decoded from the one hot encoding
plot_labels                 : to visualise different label data
validation_split            : to create training and validation datsets
split_training_data         : to split the data for cross validation
check_num_grad              : for checking the numerical gradient approximation
get_data_batch              : to get the batch data for minibtach GD
softmax                     : soft max function
eval_metrics                : calculates the accuracy given outputs
cv_fold_data                : to get the data for fold during cross validation
train_and_test              : to train on the training data and predict the labels for output data
test                        : return the accuracy of a model on a test dataset
plot                        : plotting function for visualising the data
plot_curves                 : helping function to plot all the required plots











