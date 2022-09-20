# Copyright 2022 by the author(s) of this code.
# All rights reserved.

#document my code with docstrings and comments as needed to explain what the code is doing and why.
#use the numpy library for all matrix operations



import numpy as np
import math
import random as rand
import json
import logging

#activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    return 1*(x>0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def softmax_prime(x):
    return softmax(x)*(1-softmax(x))


class _Layer:
    def __init__(self, size, activation, ):
        self.size = size
        self.activation = activation
        self.weights = np.random.rand(size, size)
        self.bias = np.random.rand(size, 1)
        self.z = np.zeros((size, 1))
        self.a = np.zeros((size, 1))
        self.delta = np.zeros((size, 1))


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, activation_function):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.activation_function = activation_function
