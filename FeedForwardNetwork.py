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


    def feedForward(self, inputs, targets):
        #passing outputs from one layer to the next
        self.activations = []
        self.dOutputs = []
        self.dCosts = []
        self.inputs = inputs
        for i, weightMatrix in zip(range(self.depth), self.weightMatrices):
            layer = self.network[i]
            inputs = layer.feedForward(inputs, weightMatrix)
            self.activations.append(inputs)
            self.dOutputs.append(layer.dReLus(inputs, weightMatrix))

        #output of output layer
        outputLayer = self.network[self.depth]
        outputs = outputLayer.outputs(inputs, self.weightMatrices[self.depth])
        self.dOutputs.append(outputLayer.dOutputs(inputs, self.weightMatrices[self.depth]))
        self.dCosts = outputLayer.dCosts(inputs, self.weightMatrices[self.depth], targets)
        return outputs

    def delta(self, l):
        if (l == self.depth):
            dCosts = numpy.array(self.dCosts)
            dSigmoids = numpy.array(self.dOutputs[l])

            return numpy.multiply(dCosts, dSigmoids)
        else:
            weightMatrix = numpy.array(self.weightMatrices[l])
            weightTranspose = weightMatrix.transpose()
            previousDelta = self.delta(l + 1)
            dReLus = numpy.array(self.dOutputs[l])

    def dWeights(self, l):
        if l = 0:
            activations = numpy.array([self.inputs])
        else:
            activations = numpy.array([self.activations[l - 1]])

        deltas = numpy.array([self.delta(l)])
        transposeDeltas = deltas.transpose()
        print(transposeDeltas)
        print(activations)
        dWeights = transposeDeltas.dot(activations)
        return dWeights
