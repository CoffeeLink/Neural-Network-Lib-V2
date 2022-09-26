import NeuralNetwork as nn

datasets = []

datasets.append(nn.DataPoint([0, 0], [0]))
datasets.append(nn.DataPoint([0, 1], [1]))
datasets.append(nn.DataPoint([1, 0], [1]))
datasets.append(nn.DataPoint([1, 1], [0]))

network = nn.NeuralNetwork([2, 3, 1], nn.sigmoidActivation())

print(network.avrageCost(datasets))

nn.logging.basicConfig(level=nn.logging.DEBUG)

network.trainWithGradientDescend(datasets, 5000, 4, 0.3)

print(network.avrageCost(datasets))

network.exportNetwork("demo_a0.nn")

print("XOR, 0, 0: " + str(network.calculateOutputs([0, 0])))
print("XOR, 1, 0: " + str(network.calculateOutputs([1, 0])))
print("XOR, 0, 1: " + str(network.calculateOutputs([0, 1])))
print("XOR, 1, 1: " + str(network.calculateOutputs([1, 1])))