import unittest
import NeuralNetwork

class Test_NeuralNetwork(unittest.TestCase):
    
    def test_random_layer(self):
        ac = NeuralNetwork.sigmoidActivation()
        layer = NeuralNetwork._Layer(2, 2, ac, ac.derivative)
        self.assertEqual(layer.inNodes, 2)
        self.assertEqual(layer.outNodes, 2)
        self.assertEqual(layer.activation, ac)
        self.assertEqual(layer.weights.shape, (2,2))
    
    def test_random_datapoint(self):
        datapoint = NeuralNetwork.DataPoint([1,2], [3,4])
        self.assertEqual(datapoint.inputs, [1,2])
        self.assertEqual(datapoint.targets, [3,4])
        self.assertEqual(datapoint.getInput(0), 1)
        self.assertEqual(datapoint.getInput(1), 2)
        self.assertEqual(datapoint.getTarget(0), 3)
        self.assertEqual(datapoint.getTarget(1), 4)
        self.assertEqual(datapoint.getInputs(), [1,2])
        self.assertEqual(datapoint.getTargets(), [3,4])
    
    def test_random_layer_cost(self):
        layer = NeuralNetwork._Layer(2, 2, NeuralNetwork.sigmoidActivation(), NeuralNetwork.sigmoidActivation())
        self.assertEqual(layer.nodeCost(1, 1), 0)
        self.assertEqual(layer.nodeCost(1, 0), 1)
        self.assertEqual(layer.nodeCost(0, 1), 1)
        self.assertEqual(layer.nodeCost(0, 0), 0)
    
    def test_gradient_descent(self):
        network = NeuralNetwork.NeuralNetwork([2,2,2], NeuralNetwork.sigmoidActivation())
        datapoint = NeuralNetwork.DataPoint([1,2], [3,4])
        network.trainWithGradientDescend([datapoint], 1, 1, 0.1)
        self.assertEqual(len(network.costVecor), 1)
       

if __name__ == '__main__':
    unittest.main()