import numpy
import Layer
from Layer import *


class FeedForwardNetwork:
    #depth specifies number of hidden layers as there is always one input layer and one output layer
    def __init__(self, depth):
        self.network = []
        self.depth = depth
        #hidden layers initialized
        for i in range(depth):
            layerSize = int(input('Layer Size: '))
            layer = Layer(layerSize)
            self.network.append(layer)

        #output layer which will likely have different activation functions
        layerSize = int(input('Output Layer Size: '))
        layer = Layer(layerSize)
        self.network.append(layer)

    def feedForward(self, inputs, weightMatrices):
        #passing outputs from one layer to the next
        for i in range(self.depth):
            inputs = self.network[i].feedForward(inputs, weightMatrix)

        #output of output layer
        outputs = self.network[depth].output(inputs, weightMatrices[depth])
        return outputs
