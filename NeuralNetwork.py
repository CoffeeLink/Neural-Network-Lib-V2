# Copyright 2022 by the author(s) of this code.
# All rights reserved.

import numpy as np
import math
import random as rand
import json
import logging

#activation functions
def linear(x):
    return x

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
    def __init__(self, inNodes, outNodes, activation):
        self.inNodes = inNodes
        self.outNodes = outNodes
        self.activation = activation
        self.weightGradient = np.zeros((inNodes, outNodes))
        self.biasGradient = np.zeros((outNodes))
        self.weights = np.random.randn(inNodes, outNodes)
        self.biases = np.zeros((outNodes))

    def calculateOutputs(self, inputs):
        weightedInputs = np.zeros((self.outNodes))

        for nodeOutput in range(self.outNodes):
            weightedInput = self.biases[nodeOutput]
            for nodeInput in range(self.inNodes):
                weightedInput += inputs[nodeInput] * self.weights[nodeInput][nodeOutput]
            weightedInputs[nodeOutput] -= self.activation(weightedInput) 
        
        return weightedInputs

    def nodeCost(self, output, target):
        error = target - output
        return error * error

    def applyGradient(self, learningRate):
        for nodeOutput in range(self.outNodes):
            for nodeInput in range(self.inNodes):
                self.weights[nodeInput][nodeOutput] = learningRate * self.weightGradient[nodeInput][nodeOutput]
            self.biases[nodeOutput] -= learningRate * self.biasGradient[nodeOutput]
    
    
    

class DataPoint:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def getInput(self, index):
        return self.inputs[index]
    
    def getTarget(self, index):
        return self.targets[index]

    def getInputs(self):
        return self.inputs

    def getTargets(self):
        return self.targets


class NeuralNetwork:
    def __init__(self, layer_sizes : list, activation_function):
        self.layer_sizes = layer_sizes
        self.layers = [] #list of layers
        for i in range(len(layer_sizes)-1):
            self.layers.append(_Layer(layer_sizes[i], layer_sizes[i+1], activation_function)) #create layers
        self.activation_function = activation_function
        
    def calculateOutputs(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.calculateOutputs(outputs)
        return outputs

    def cost(self, dataPoint : DataPoint):
        outputs = self.calculateOutputs(dataPoint.getInputs())
        cost = 0
        for node in range(len(outputs)):
            cost += self.layers[-1].nodeCost(outputs[node], dataPoint.getTarget(node))
        return cost
    
    def avrageCost(self, data : list):
        cost = 0
        for dataPoint in data:
            cost += self.cost(dataPoint)
        return cost / len(data)
    
    def gradientDescent(self, data : list, learningRate = 0.1):
        h = 0.0001
        
        originalCost = self.avrageCost(data)
        
        for layer in self.layers:
            for nodeInput in range(layer.inNodes):
                for nodeOutput in range(layer.outNodes):
                    layer.weights[nodeInput][nodeOutput] += h
                    deltaCost = self.avrageCost(data) - originalCost
                    layer.weights[nodeInput][nodeOutput] -= h
                    layer.weightGradient[nodeInput][nodeOutput] = deltaCost / h
            
            for nodeOutput in range(layer.outNodes):
                layer.biases[nodeOutput] += h
                deltaCost = self.avrageCost(data) - originalCost
                layer.biases[nodeOutput] -= h
                layer.biasGradient[nodeOutput] = deltaCost / h
            
        for layer in self.layers:
            layer.applyGradient(learningRate)
        
    def train(self, data : list, epochs, batch_size,learningRate = 0.1):
        for i in range(epochs):
            rand.shuffle(data)
            batches = [data[k:k+batch_size] for k in range(0, len(data), batch_size)]
            for batch in batches:
                self.gradientDescent(batch, learningRate)
            if i % 100 == 0:
                print("Epoch: " + str(i) + " Cost: " + str(self.avrageCost(data)))
            

n = NeuralNetwork([2,3,2], sigmoid)

dataSet = []
dataSet.append(DataPoint([0,0], [0,0]))
dataSet.append(DataPoint([0,1], [1,0]))
dataSet.append(DataPoint([1,0], [1,0]))
dataSet.append(DataPoint([1,1], [0,1]))

n.train(dataSet, 10000, 1, 0.1)

print(n.calculateOutputs([1,1]))