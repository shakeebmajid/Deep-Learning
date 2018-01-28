import numpy
import Layer
import Neuron
from Layer import *


class FeedForwardNetwork:

    #depth specifies number of hidden layers as there is always one input layer and one output layer
    def __init__(self, depth):
        self.network = []
        self.depth = depth
        self.weightMatrices = []
        #input size used for initializing first weight matrix
        inputSize = int(input('Input Size: '))
        #hidden layers initialized
        for i in range(depth):
            layerSize = int(input('Layer Size: '))
            weightMatrix = numpy.random.rand(layerSize, inputSize)
            self.weightMatrices.append(weightMatrix)
            layer = Layer(layerSize)
            self.network.append(layer)
            inputSize = layerSize

        #output layer which will likely have different activation functions
        layerSize = int(input('Output Layer Size: '))
        layer = Layer(layerSize)
        self.network.append(layer)

        weightMatrix = numpy.random.rand(layerSize, inputSize)
        self.weightMatrices.append(weightMatrix)


    def feedForward(self, inputs):
        #passing outputs from one layer to the next
        for i, weightMatrix in zip(range(self.depth), self.weightMatrices):
            inputs = self.network[i].feedForward(inputs, weightMatrix)

        #output of output layer
        outputs = self.network[depth].output(inputs, self.weightMatrices[depth])

        return outputs

    def error(self, inputs, weightMatrix, targets):
        outputLayer = self.network[self.depth]
        dCosts = numpy.array(outputLayer.dCosts(inputs, weightMatrix, targets))
        dSigmoids = numpy.array(outputLayer.dSigmoids(inputs, weightMatrix, targets))

        return numpy.multiply(dCosts, dSigmoids)


    def error()
