#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# @author: Sang-hyeb Lee
# @license: MIT

"""Implements rudimentary deep neural network in Python3

DeepNeuralNetwork class implements a minimal implementation of deep neural network.
"""

import numpy as np

# utility function
class Utility:
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_backward(dA, Z):
        relu_fn = lambda x: 0 if x <= 0 else 1
        return np.vectorize(relu_fn)(Z) * dA

    @staticmethod
    def sigmoid_backward(dA, Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ

#worker class
class DeepNeuralNetwork:
    """implements a deep neural network"""

    def __init__(self,n,activation_list):
        self.initialize_parameters(n,activation_list)

    def predict(self,sample):
        AL, caches = self.perform_forward_propagation(sample, self._parameters)
        return AL

    def train(self,dataset, hyperparameters=None):
        """train network for given dataset"""
        # 1. Define hyperparameters if not given
        if hyperparameters is None:
            hyperparameters={}
            hyperparameters["learning_rate"] = 0.0075
            hyperparameters["num_iteration"] = 100

        X = dataset["X"]
        Y = dataset["Y"]
        # 2. Loop for num_iterations:
        for i in range(hyperparameters["num_iteration"]):
            #(a) perform forward propagation
            AL, caches = self.perform_forward_propagation(X, self._parameters)

            #(b) compute cost function
            cost = self.compute_cost_binary(AL,Y)

            #(c) perform backward propagation
            grads = self.perform_backward_propagation(AL,dataset["Y"],caches,self._parameters)

            #(d) perform update
            self._parameters = self.update_parameteres(self._parameters,grads,hyperparameters["learning_rate"])

            #(4) print the cost
            print(f"cost at {i}th iteration : {cost}")

    def initialize_parameters(self, n, activation_list):
        """
        @parameters:
        n -- a list containing the dimensions of each layer in our network

        @returns:
        parameters -- a dictionary containing w,b for each layer
        """
        # (a) initialize structures for parameters
        L = len(n)  # number of layers
        parameters = {}  # using dictionaray struct
        parameters["W"] = {}
        parameters["b"] = {}
        parameters["L"] = L
        parameters["activation"] = activation_list

        # (b) initialize parameters for each layer
        for l in range(1, L):
            # need to initialize W to random number or each hidden layer will compute the same number
            # size of W matrix = (n[l],n[l-1]), n_l = number of units in lth layer
            # size of b column vector = (n_l,1)
            parameters["W"][l] = np.random.randn(n[l], n[l - 1]) * 0.01
            parameters['b'][l] = np.zeros((n[l], 1))

        # return the initialized values
        self._parameters = parameters
        return parameters

    @staticmethod
    def linear_activation_forward(A_prev, W, b, activation):
        """Perform forward propagation with given activation function

        @parameters:
        A_prev = activation matrix from previous layer (n_l-1, m)
        W = weight matrix (n_l,n_1)
        b = bias vector

        @returns
        """
        # (a) perform linear part
        Z = np.dot(W, A_prev) + b

        # (b) perform non-linear activation
        if activation == "sigmoid":
            A = Utility.sigmoid(Z)
        elif activation == "relu":
            A = Utility.relu(Z)
        else:  # just linear activation
            A = Z

        assert (A.shape == (W.shape[0], A_prev.shape[1]))

        cache = ((A_prev,W,b),Z)
        return A, cache

    def perform_forward_propagation(self,X,parameters):
        """ Implement forwad propagation

        @parameters:
        X -- input feature vector (number of features, number of examples)
        parameters -- contains parameters for each layer
        """

        A = X
        L = parameters["L"]
        caches = []

        # Forward propagate!
        for l in range(1, L):
            A_prev = A  # input from the previous layer
            W = parameters["W"][l]
            b = parameters["b"][l]
            activation = parameters["activation"][l]
            A, cache = self.linear_activation_forward(A_prev, W, b, activation)
            caches.append(cache)

        return A, caches

    #   b. compute cost function
    def compute_cost_binary(self, AL, Y):
        """Compute the cost function(x) = -1/m * np.sum(Y * np.log(AL)+(1-Y)*np.log(1-AL))

        @parameters:
        AL -- our current prediction vector (1, num of examples)
        Y -- true prediction vector (1 number of examples)
        """

        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        cost = np.squeeze(cost)  # turn [[]]] into scalar

        return cost

    @staticmethod
    def linear_activation_backward(dA, cache, activation):
        """Perform backward propagation

        @parameters
        dA -- activation gradient for the current layer
        cache -- cache
        activation -- which activation

        @returns

        """

        linear_cache, Z = cache  # activation_cache = z
        # (1) perform non-linear(activation) part to workout dZ
        #
        # dZ = (da) * (g'(Z))
        if activation == "relu":
            dZ = Utility.relu_backward(dA, Z)
        elif activation == "sigmoid":
            dZ = Utility.sigmoid_backward(dA, Z)


        # (2) perform linear part
        A_prev, W, b = linear_cache
        m = A_prev.shape[1]

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, keepdims=True, axis=1)
        dA_prev = np.dot(W.T, dZ)

        #sanity check
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def perform_backward_propagation(self, AL, Y, caches, parameters):
        """Perform backward propagation for given network

        @parameters
        AL -- our current guess
        Y  -- true label vector
        caches -- cached information
        parameters -- parameters

        @returns
        gradients -- computed gradients for dA, dW, db
        """

        # (1) prepare data
        L = len(caches)
        m = AL.shape[1]  # number of examples
        Y = Y.reshape(AL.shape)  # same shape as AL

        gradients = {}
        gradients["dA"] = {}
        gradients["dW"] = {}
        gradients["db"] = {}

        # (2) perform backpropagation
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # element-wise division

        for l in reversed(range(L)):  # from L-1 to 0

            current_cache = caches[l]
            dA_prev, dW, db = self.linear_activation_backward(dAL, current_cache, parameters["activation"][l + 1])
            # store the output
            # layer 1 - hidden layer
            # layer 0 - input layer
            gradients["dA"][l] = dA_prev  # input for previous layer
            gradients["dW"][l + 1] = dW  # for the current layer.
            gradients["db"][l + 1] = db  # for the current layer

            # for next
            dAL = dA_prev

        # return the result
        return gradients

    #   d. update parameters using gradient descent
    def update_parameteres(self, parameters, grads, learning_rate):
        """Update parameters using gradient descent

        @parameters
        parameters -- contains W,b for each layer
        grads -- gradients
        learning_rate -- learning rate alpha

        @returns
        """
        L = parameters["L"]
        #print(grads)
        for l in range(1,L):

            parameters["W"][l] = parameters["W"][l] - learning_rate * grads["dW"][l]
            parameters["b"][l] = parameters["b"][l] - learning_rate * grads["db"][l]

        return parameters
