from operator import ne
import NeuralNetwork
import random

datasets = []

network = NeuralNetwork.NeuralNetwork([9, 4, 4, 4], NeuralNetwork.sigmoidActivation())

cube = [[1,1,1], [1,0,1], [1,1,1]]
cubeFull = [[1,1,1], [1,1,1], [1,1,1]]

line1 = [[1,1,1], [0,0,0], [0,0,0]]
line2 = [[0,0,0], [1,1,1], [0,0,0]]
line3 = [[0,0,0], [0,0,0], [1,1,1]]

line4 = [[1,0,0], [1,0,0], [1,0,0]]
line5 = [[0,1,0], [0,1,0], [0,1,0]]
line6 = [[0,0,1], [0,0,1], [0,0,1]]

plus = [[0,1,0], [1,1,1], [0,1,0]]

cross = [[1,0,0], [0,1,0], [0,0,1]]
cross2 = [[0,0,1], [0,1,0], [1,0,0]]

def addNoise(obj, noiseRangeFrom, noiseRangeTo):
    for i in range(len(obj)-1):
        for pi in range(len(obj[i])-1):
            pixel = obj[i][pi]
            factor = random.uniform(noiseRangeFrom, noiseRangeTo)
            pixel += factor
            obj[i][pi] = pixel
    return obj
        
            

cubes = []
lines = []
pluses = []
crosses = []

for i in range(10):
    cubes.append(addNoise(cube, -0.2, 0.2))
    cubes.append(addNoise(cubeFull, -0.2, 0.2))
    lines.append(addNoise(line1, -0.2, 0.2))
    lines.append(addNoise(line2, -0.2, 0.2))
    lines.append(addNoise(line3, -0.2, 0.2))
    lines.append(addNoise(line4, -0.2, 0.2))
    lines.append(addNoise(line5, -0.2, 0.2))
    lines.append(addNoise(line6, -0.2, 0.2))
    pluses.append(addNoise(plus, -0.2, 0.2))
    crosses.append(addNoise(cross, -0.2, 0.2))
    crosses.append(addNoise(cross2, -0.2, 0.2))

for item in cubes:
    datasets.append(NeuralNetwork.DataPoint([item[0][0], item[0][1], item[0][2], item[1][0], item[1][1], item[1][2], item[2][0], item[2][1], item[2][2]], [1,0,0,0]))

for item in lines:
    datasets.append(NeuralNetwork.DataPoint([item[0][0], item[0][1], item[0][2], item[1][0], item[1][1], item[1][2], item[2][0], item[2][1], item[2][2]], [0,1,0,0]))

for item in pluses:
    datasets.append(NeuralNetwork.DataPoint([item[0][0], item[0][1], item[0][2], item[1][0], item[1][1], item[1][2], item[2][0], item[2][1], item[2][2]], [0,0,1,0]))

for item in crosses:
    datasets.append(NeuralNetwork.DataPoint([item[0][0], item[0][1], item[0][2], item[1][0], item[1][1], item[1][2], item[2][0], item[2][1], item[2][2]], [0,0,0,1]))

NeuralNetwork.logging.basicConfig(level=NeuralNetwork.logging.DEBUG)

network.trainWithGradientDescend(datasets, 3500, len(datasets), 0.2)
print(network.avrageCost(datasets))
network.exportNetwork("recog01.json")