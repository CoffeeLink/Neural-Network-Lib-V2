# Copyright 2022 by the author(s) of this code.
# All rights reserved.


from threading import Thread
import numpy as np
import random as rand
import logging
import json


#activation functions
class ActivationFunction:
    def __init__(self, activation, derivative, activation_id):
        """ActivationFunction is a class that holds an activation function and its derivative.

        Args:
            activation (function): The activation function.
            derivative (function): The derivative of the activation function.
        """
        self.activation = activation
        self.derivative = derivative
        self.activation_id = activation_id
    
    def __call__(self, x : float):
        """__call__ is a function that calls the activation function.
        
        Args:
            x (float): The input to the activation function.
        
        Returns:
            float: The output of the activation function.
        """
        return self.activation(x)
    
    def derivative(self, x : float):
        """derivative is a function that calls the derivative of the activation function.

        Args:
            x (float): The input to the derivative of the activation function.

        Returns:
            float: The output of the derivative of the activation function.
        """
        return self.derivative(x)

class sigmoidActivation(ActivationFunction):
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    
    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_derivative, 1)

class linearActivation(ActivationFunction):
    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return 1
    
    def __init__(self):
        super().__init__(self.linear, self.linear_derivative, 0)

class tanhActivation(ActivationFunction):
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1.0 - x**2
    def __init__(self):
        super().__init__(self.tanh, self.tanh_derivative, 2)

class reluActivation(ActivationFunction):
    def relu(self, x):
        return np.maximum(0,x)

    def relu_derivative(self, x):
        return 1*(x>0)
    
    def __init__(self):
        super().__init__(self.relu, self.relu_derivative, 3)

class softmaxActivation(ActivationFunction):
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)

    def softmax_derivative(self, x):
        return self.softmax(x)*(1-self.softmax(x))
    
    def __init__(self):
        super().__init__(self.softmax, self.softmax_derivative, 5)

class _Layer:
    def __init__(self, inNodes : int, outNodes : int, activation, activationDerivative, threadLimit : int = 1):
        """Initalize a new layer with the given number of nodes and activation function
        
        Arguments:
            inNodes [int] -- number of nodes in the previous layer
            outNodes [int] -- number of nodes in this layer
            activation [function] -- activation function for this layer
            activationDerivative [function] -- derivative of the activation function
        """
        self.inNodes = inNodes
        self.outNodes = outNodes
        self.activation = activation
        self.activationDerivative = activationDerivative
        
        self.activations = np.zeros((outNodes))
        self.weightedInputs = np.zeros((outNodes))
        self.inputs = np.zeros((inNodes))
        
        self.weightGradient = np.zeros(np.shape([threadLimit, inNodes, outNodes]))
        self.biasGradient = np.zeros((threadLimit, outNodes))
        self.threadLimit = threadLimit
        
        self.weights = np.random.randn(inNodes, outNodes)
        self.biases = np.zeros((outNodes))

    def calculateOutputs(self, inputs : list[float]):
        """Calculate the outputs of this layer
        
        Arguments:
            inputs {list[float]} -- inputs to this layer
        
        Returns:
            list[float] -- outputs of this layer
        """
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

    def applyGradient(self, learningRate : float):
        """Apply the gradient to the weights and biases
        
        Arguments:
            learningRate {float} -- learning rate
        """
        for thread in range(self.threadLimit):
            for nodeOutput in range(self.outNodes):
                for nodeInput in range(self.inNodes):
                    self.weights[nodeInput][nodeOutput] -= learningRate * self.weightGradient[thread][nodeInput][nodeOutput]
                self.biases[nodeOutput] -= learningRate * self.biasGradient[thread][nodeOutput]
    
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
    def __init__(self, inputs : list[float], targets : list[float]):
        """A single with inputs and targets

        Args:
            inputs (List[float]): The inputs of the data point
            targets (List[float]): The targets to achieve
        """
        self.inputs = inputs
        self.targets = targets
    
    def getInput(self, index : int):
        """Get the input at the given index

        Args:
            index (int): The index of the input

        Returns:
            float: The input at the given index
        """
        return self.inputs[index]
    
    def getTarget(self, index):
        """Get the target at the given index

        Args:
            index (int): The index of the target

        Returns:
            float: The target at the given index
        """
        return self.targets[index]

    def getInputs(self):
        """"Get the inputs of the data point

        Returns:
            List[float]: The inputs of the data point
        """
        return self.inputs

    def getTargets(self):
        """Get the targets of the data point

        Returns:
            List[float]: The targets of the data point
        """
        return self.targets


class NeuralNetwork:
    def __init__(self, layer_sizes : list[int], activation_function : ActivationFunction, threadLimit : int = 1):
        """Creates a new neural network with the given layer sizes and activation functions

        Args:
            layer_sizes (list[int]): The sizes of the layers in the network
            activation_function (ActivationFunction): The activation function to use for the network (same for all layers)
        """
        self.layer_sizes = layer_sizes
        self.known_activation_functions = {0: linearActivation(), 1: sigmoidActivation(), 2: tanhActivation(), 3: reluActivation(), 5: softmaxActivation()} # a dict for the activation functions to use when importing a network
        self.layers = [] #list of layers
        self.costVecor = [] #list of costs
        for i in range(len(layer_sizes)-1):
            self.layers.append(_Layer(layer_sizes[i], layer_sizes[i+1], activation_function, activation_function.derivative, threadLimit=threadLimit)) #create layers
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function.derivative
        self.l0NumNodesIn = layer_sizes[0]
        self.l0NumNodesOut = layer_sizes[1] # the number of nodes in the 2. layer
        self.threadLimit = threadLimit
        
    def calculateOutputs(self, inputs: list):
        """Calculates the outputs of the network for the given inputs

        Args:
            inputs (List): The input values for the network

        Returns:
            List[float]: The output values of the network
        """
        outputs = inputs
        for layer in self.layers:
            outputs = layer.calculateOutputs(outputs)
        return outputs

    def nodeCost(self, output, target):
        return (output - target)**2
    
    def nodeCostDerivative(self, output, target):
        return 2*(output - target)
    
    def cost(self, dataPoint : DataPoint):
        """Calculates the cost of the network for the given data point

        Args:
            dataPoint (DataPoint): The data point to calculate the cost for

        Returns:
            float: The cost of the network for the given data point
        """
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
    
    def avrageCost(self, data : list[DataPoint]):
        """Calculate the avrage cost of the network based on given data
        
        Args:
            data (list[DataPoint]): list of data points to calculate the avrage cost of
        Returns:
            float: avrage cost of the network
        """
        cost = 0
        for dataPoint in data:
            cost += self.cost(dataPoint)
        return cost / len(data)
    
    def applyGradients(self, learningRate):
        """Applies the gradients to the weights and biases
        
        Args:
            learningRate (float): The learning rate
        """
        for layer in self.layers:
            layer.applyGradient(learningRate)
    
    def clearGradients(self):
        """Clears the gradients of the weights and biases"""
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
    
    def learn(self, dataPoints : list[DataPoint], iterations : int, learningRate=0.1):
        """[WIP] Trains the neural network with backpropagation on the given data points
        
        Args:
            dataPoints (list[DataPoint]): The data points to train on
            iterations (int): The number of iterations to train for
            learningRate (float, optional): The learning rate. Defaults to 0.1.
        """
        for i in range(iterations):
            self._learn(dataPoints, learningRate)
            if i % 100 == 0:
                print("Iteration: " + str(i) + " Cost: " + str(self.avrageCost(dataPoints)))
    
    def gradientDescent(self, data : list, learningRate = 0.1, threadId = 0):
        h = 0.0001
        self.clearGradients()
        
        originalCost = self.avrageCost(data)
        
        for layer in self.layers:
            for nodeInput in range(layer.inNodes):
                for nodeOutput in range(layer.outNodes):
                    layer.weights[nodeInput][nodeOutput] += h
                    deltaCost = self.avrageCost(data) - originalCost
                    layer.weights[nodeInput][nodeOutput] -= h
                    layer.weightGradient[threadId][nodeInput][nodeOutput] = deltaCost / h
            
            for nodeOutput in range(layer.outNodes):
                layer.biases[nodeOutput] += h
                deltaCost = self.avrageCost(data) - originalCost
                layer.biases[nodeOutput] -= h
                layer.biasGradient[threadId][nodeOutput] = deltaCost / h

        self.applyGradients(learningRate)
    
    def trainWithGradientDescend(self, data : list[DataPoint], epochs : int, batch_size : int ,learningRate = 0.1):
        """Trains the network with gradient descend

        Args:
            data (list[DataPoint]): list of data points to train with
            epochs (int): number of epochs to train
            batch_size (int): size of the batch to train with each epoch
            learningRate (float, optional): The rate the network learns with. Defaults to 0.1.
        """
        for i in range(epochs):
            rand.shuffle(data)
            batches = [data[k:k+batch_size] for k in range(0, len(data), batch_size)]
            for batch in batches:
                self.gradientDescent(batch, learningRate)
            if i % 100 == 0:
                self.costVecor.append([self.avrageCost(data), i])
                logging.debug("Epoch: " + str(i) + " Cost: " + str(self.avrageCost(data)))
    
    def trainWithGradientDescendMultiThread(self, data : list[DataPoint], epochs : int, batch_size : int ,learningRate = 0.1, threadCount = 1):
        """Trains the network with gradient descend

        Args:
            data (list[DataPoint]): list of data points to train with
            epochs (int): number of epochs to train
            batch_size (int): size of the batch to train with each epoch
            learningRate (float, optional): The rate the network learns with. Defaults to 0.1.
        """
        for i in range(epochs):
            rand.shuffle(data)
            batches = [data[k:k+batch_size] for k in range(0, len(data), batch_size)]
            for batch in batches:
                threads = []
                for threadId in range(threadCount):
                    threads.append(Thread(target=self.gradientDescent, args=(batch, learningRate, threadId)))
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            if i % 100 == 0:
                self.costVecor.append([self.avrageCost(data), i])
                logging.debug("Epoch: " + str(i) + " Cost: " + str(self.avrageCost(data)))
    
    def exportNetwork(self, fileName : str):
        """Saves the network to a file

        Args:
            fileName (str): name of the file to save to
        """
        with open(fileName, "w+") as f:
            f.truncate(0)
            parsed = {"layers" : [], "layerSizes" : self.layer_sizes}
            i = 0
            w = np.zeros((self.l0NumNodesIn, self.l0NumNodesOut)).tolist()
            b = np.zeros(self.l0NumNodesOut).tolist()
            parsed["layers"].append({"weights" : w, "biases" : b, "isInputLayer" : True, "activationType": 0})
            for layer in self.layers:
                parsed["layers"].append({"weights" : layer.weights.tolist(), "biases" : layer.biases.tolist(), "isInputLayer" : False, "activationType": layer.activation.activation_id})
            f.write(json.dumps(parsed, indent=4, sort_keys=True))
                
    
    def importNetwork(self, fileName : str):
        """loads a network from a file
        
        Args:
            fileName (str): name of the file to load from
        """
        with open(fileName, "r") as f:
            parsed = json.load(f)
            layers = []
            for layer in parsed["layers"]:
                if not layer["isInputLayer"]:
                    layerC = _Layer(inNodes=len(layer["weights"]), outNodes=len(layer["biases"]), activation=self.known_activation_functions[layer["activationType"]], activationDerivative=self.known_activation_functions[layer["activationType"]].derivative)
                    layerC.weights = np.array(layer["weights"])
                    layerC.biases = np.array(layer["biases"])
                    layers.append(layerC)
                else:
                    self.layer_sizes[0] = len(layer["biases"])
            self.layers = layers
