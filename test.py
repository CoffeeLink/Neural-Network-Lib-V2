import NeuralNetwork as nn
network = nn.NeuralNetwork([2, 2, 1], nn.sigmoidActivation())

data = []
data.append(nn.DataPoint([0, 0], [0]))
data.append(nn.DataPoint([0, 1], [1]))
data.append(nn.DataPoint([1, 0], [1]))
data.append(nn.DataPoint([1, 1], [0]))

for i in range(10000):
    network.trainOnline(data, 0.1)
    print(network.avrageCost(data))