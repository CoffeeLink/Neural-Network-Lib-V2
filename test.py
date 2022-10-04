import NeuralNetwork as nn
#Path: NeuralNetwork.py

network = nn.NeuralNetwork([2, 2, 1], nn.sigmoidActivation(), threadLimit=10)

data = []
data.append(nn.DataPoint([0, 0], [0]))
data.append(nn.DataPoint([0, 1], [1]))
data.append(nn.DataPoint([1, 0], [1]))
data.append(nn.DataPoint([1, 1], [0]))

network.trainWithGradientDescendMultiThread(data, 1000, 4, 0.1, threadCount=10)

print(network.calculateOutputs([0, 0]))