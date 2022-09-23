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

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - x**2

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return 1*(x>0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def softmax_derivative(x):
    return softmax(x)*(1-softmax(x))


class _Layer:
    def __init__(self, inNodes, outNodes, activation, activationDerivative):
        self.inNodes = inNodes
        self.outNodes = outNodes
        self.activation = activation
        self.activationDerivative = activationDerivative
        
        self.activations = np.zeros((outNodes))
        self.weightedInputs = np.zeros((outNodes))
        self.inputs = np.zeros((inNodes))
        
        self.weightGradient = np.zeros((inNodes, outNodes))
        self.biasGradient = np.zeros((outNodes))
        
        self.weights = np.random.randn(inNodes, outNodes)
        self.biases = np.zeros((outNodes))

    def calculateOutputs(self, inputs):
        weightedInputs = np.zeros((self.outNodes))
        self.inputs = inputs

        for nodeOutput in range(self.outNodes):
            weightedInput = self.biases[nodeOutput]
            for nodeInput in range(self.inNodes):
                weightedInput += inputs[nodeInput] * self.weights[nodeInput][nodeOutput]
            self.weightedInputs[nodeOutput] = weightedInput
            weightedInputs[nodeOutput] = self.activation(weightedInput) 

        self.activations = weightedInputs
        return weightedInputs

    def applyGradient(self, learningRate):
        for nodeOutput in range(self.outNodes):
            for nodeInput in range(self.inNodes):
                self.weights[nodeInput][nodeOutput] = learningRate * self.weightGradient[nodeInput][nodeOutput]
            self.biases[nodeOutput] -= learningRate * self.biasGradient[nodeOutput]
    
    def clearGradients(self):
        self.weightGradient = np.zeros((self.inNodes, self.outNodes))
        self.biasGradient = np.zeros((self.outNodes))
    
    def nodeCost(self, output, target):
        return (output - target)**2
    
    def nodeCostDerivative(self, output, target):
        return 2*(output - target)
    
    def calculateOutputLayerValues(self, expectedOutputs):
        nodeValues = np.zeros((self.outNodes))
        for node in range(self.outNodes):
            costDerivative = self.nodeCostDerivative(self.activations[node], expectedOutputs[node])
            activationDerivative = self.activationDerivative(self.activations[node])
            nodeValues[node] = costDerivative * activationDerivative
        
        return nodeValues

    def updateGradients(self, nodeValues):
        for nodeOutput in range(self.outNodes):
            for nodeInput in range(self.inNodes):
                derivativeCostWithRespectToWeight = nodeValues[nodeOutput] * self.inputs[nodeInput]
                self.weightGradient[nodeInput][nodeOutput] += derivativeCostWithRespectToWeight
            derivativeCostWithRespectToBias = nodeValues[nodeOutput]
            self.biasGradient += derivativeCostWithRespectToBias
    
    def calculateHiddenLayerValues(self, oldLayer, oldNodeValues):
        newNodeValues = np.zeros((self.outNodes))
        for node in range(self.outNodes):
            newNodeValue = 0
            for oldNode in range(oldLayer.outNodes):
                weightedInputDerivate = oldLayer.weights[node][oldNode]
                newNodeValue += oldNodeValues[oldNode] * weightedInputDerivate
            newNodeValue *= self.activationDerivative(self.activations[node])
            newNodeValues[node] = newNodeValue
        
        return newNodeValues
                
            
    

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
    def __init__(self, layer_sizes : list, activation_function, activation_function_derivative):
        self.layer_sizes = layer_sizes
        self.layers = [] #list of layers
        for i in range(len(layer_sizes)-1):
            self.layers.append(_Layer(layer_sizes[i], layer_sizes[i+1], activation_function, activation_function_derivative)) #create layers
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        
    def calculateOutputs(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer.calculateOutputs(outputs)
        return outputs

    def nodeCost(self, output, target):
        return (output - target)**2
    
    def nodeCostDerivative(self, output, target):
        return 2*(output - target)
    
    def cost(self, dataPoint : DataPoint):
        outputs = self.calculateOutputs(dataPoint.getInputs())
        cost = 0
        for i in range(len(outputs)):
            cost += self.nodeCost(outputs[i], dataPoint.getTarget(i))
        return cost
    
    def costDerivative(self, dataPoint : DataPoint):
        outputs = self.calculateOutputs(dataPoint.getInputs())
        costDerivative = []
        for i in range(len(outputs)):
            costDerivative.append(2*(outputs[i] - dataPoint.getTarget(i)))
        return costDerivative
    
    def avrageCost(self, data : list):
        cost = 0
        for dataPoint in data:
            cost += self.cost(dataPoint)
        return cost / len(data)
    
    def applyGradients(self, learningRate):
        for layer in self.layers:
            layer.applyGradient(learningRate)
    
    def clearGradients(self):
        for layer in self.layers:
            layer.clearGradients()
    
    def updateAllGradients(self, dataPoint : DataPoint):
        self.calculateOutputs(dataPoint.getInputs())
        outLayer = self.layers[-1]
        nodeValues = outLayer.calculateOutputLayerValues(dataPoint.getTargets())
        outLayer.updateGradients(nodeValues)
        for i in range(len(self.layers)-2, -1, -1):
            nodeValues = self.layers[i].calculateHiddenLayerValues(self.layers[i+1], nodeValues)
            self.layers[i].updateGradients(nodeValues)
            
    def _learn(self, dataPoints : list, learningRate):
        for dataPoint in dataPoints:
            self.updateAllGradients(dataPoint)
        self.applyGradients(learningRate / len(dataPoints))
        self.clearGradients()
    
    def learn(self, dataPoints : list, learningRate, iterations):
        for i in range(iterations):
            self._learn(dataPoints, learningRate)
            if i % 100 == 0:
                print("Iteration: " + str(i) + " Cost: " + str(self.avrageCost(dataPoints)))
    
    
    
n = NeuralNetwork([2,3,2], sigmoid, sigmoid_derivative)

dataSet = []
dataSet.append(DataPoint([0,0], [1,0]))
dataSet.append(DataPoint([0,1], [1,0]))
dataSet.append(DataPoint([1,0], [1,0]))
dataSet.append(DataPoint([1,1], [0,1]))

n.learn(dataSet, 0.1, 1000)

print(n.calculateOutputs([1,1]))