import NeuralNetwork as nn
import numpy as np
import time
# Path: NeuralNetwork.py

network = nn.NeuralNetwork([96, 30, 10, 2], nn.sigmoidActivation())

st = time.time()
network.calculateOutputs(np.zeros(96))
print((time.time() - st)*1000)
