import NeuralNetwork

network = NeuralNetwork.NeuralNetwork([2, 3, 1], NeuralNetwork.sigmoidActivation())

dataSets = []
dataSets.append(NeuralNetwork.DataPoint([0, 0], [0]))
dataSets.append(NeuralNetwork.DataPoint([0, 1], [1]))
dataSets.append(NeuralNetwork.DataPoint([1, 0], [1]))
dataSets.append(NeuralNetwork.DataPoint([1, 1], [0]))

test = []
test.append(NeuralNetwork.DataPoint([0, 0], [0]))
test.append(NeuralNetwork.DataPoint([0, 0.9], [1]))
test.append(NeuralNetwork.DataPoint([0.9, 0], [1]))
test.append(NeuralNetwork.DataPoint([1, 0.6], [0]))

network.trainWithGradientDescend(dataSets, 10000, len(dataSets), 0.01, test, True)