from turtle import width
import NeuralNetwork as nn
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from PIL import Image
 

# Path: NeuralNetwork.py

network = nn.NeuralNetwork([2, 10, 10, 2], nn.sigmoidActivation())

plt.title("Térkép")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")

img = Image.open("src image.jpg")
width, height = img.size

image = mpimg.imread("src image.jpg")
plt.imshow(image)

print("Width: ", width)
print("Height: ", height)

dataSets = []
falseCount = 0
trueCount = 0
while trueCount < 50 or falseCount < 50:
    w = np.random.randint(0, width)
    h = np.random.randint(0, height)
    if w >= 1100 and w <= 1600 and h >= 600 and h <= 800 and trueCount < 50:
        dataSets.append(nn.DataPoint([w, h], [1, 0]))
        trueCount += 1
    elif falseCount < 50:
        dataSets.append(nn.DataPoint([w, h], [0, 1]))
        falseCount += 1

st = time.time()
network.trainWithGradientDescend(dataSets, 1, len(dataSets), 0.3)
#network.exportNetwork("network.json")
print("Training finished")
print("Time: ", str(time.time() - st) + "s")
print("Starting Render...")

# Visualize the image pixels as a scatter plot
st = time.time()
for w in range(width):
    for h in range(height):
        if w % 20 == 0 and h % 20 == 0:
            out = network.calculateOutputs([w, h])
            if out[0] > out[1]:
                plt.plot(w, h, 'ro', markersize=2)
            else:
                plt.plot(w, h, 'bo', markersize=2)
            #plt.pause(0.0001)
for d in dataSets:
    if d.targets[0] == 1:
        plt.plot(d.inputs[0], d.inputs[1], 'go', markersize=5)
print("Render Complete!\nTime: ", str(time.time() - st) + "s")
plt.show()