import numpy
import Layer
import Neuron
import random
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
        #print("weight matrices: ", self.weightMatrices)
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
        self.activations.append(outputs)
        self.dOutputs.append(outputLayer.dOutputs(inputs, self.weightMatrices[self.depth]))
        self.dCosts = outputLayer.dCosts(inputs, self.weightMatrices[self.depth], targets)

        self.cost = self.totalCost(outputLayer, inputs, self.weightMatrices[self.depth], targets)
        return outputs

    def totalCost(self, layer, inputs, weightMatrix, targets):
        costs = layer.costs(inputs, weightMatrix, targets)
        totalCost = numpy.sum(costs)
        return totalCost

    def delta(self, l):
        if (l == self.depth):
            # print("l: ", l)
            dCosts = numpy.array(self.dCosts)
            dSigmoids = numpy.array(self.dOutputs[l])

            return numpy.multiply(dCosts, dSigmoids)
        else:
            # print("l: ", l)
            previousWeightMatrix = numpy.array(self.weightMatrices[l + 1])
            previousWeightTranspose = previousWeightMatrix.transpose()
            previousDelta = self.delta(l + 1)
            dReLus = numpy.array(self.dOutputs[l])
            # print("dRelus: ", dReLus)
            # print("transpose: ", previousWeightTranspose)
            # print("previous delta: ", previousDelta)
            return numpy.multiply(previousWeightTranspose.dot(previousDelta), dReLus)



    def dWeights(self, l):
        if l == 0:
            activations = numpy.array([self.inputs])
        else:
            activations = numpy.array([self.activations[l - 1]])

        transposeDeltas = numpy.array([self.delta(l)]).transpose()

        #print("transpose: ", transposeDeltas)
        #print("activations: ", activations)
        dWeights = transposeDeltas.dot(activations)
        #print("dWeights: ", dWeights)
        return dWeights

    def backProp(self, learningRate):
        network = self.network
        for i, layer in reversed(list(enumerate(network))):
            # print i, "weights: ", self.weightMatrices[i], "dCost / dWeights: ", self.dWeights(i)
            #print("pre: ", self.weightMatrices[i])
            #print("pre: ", self.weightMatrices)
            self.weightMatrices[i] = self.weightMatrices[i] - learningRate * self.dWeights(i)
            #print("change: ", self.weightMatrices[i] - learningRate * self.dWeights(i))
            #print("post: ", self.weightMatrices[i])
            #print("post: ", self.weightMatrices)

            #print self.dWeights(i)

    def trainItem(self, inputs, targets, learningRate):
        self.feedForward(inputs, targets)
        self.backProp(learningRate)

    def train(self, trainingSize, learningRate):
        #write
        f = open("XOR-training.txt", "w+")
        for i in range(trainingSize):
            a = bool(random.getrandbits(1))
            b = bool(random.getrandbits(1))
            f.write (str(int(a)) + " " + str(int(b)) + " " + str(int(a ^ b)) + "\n")
            #f.write("This is line %d\r\n" % (i+1))

        f.close()


        alist = [line.rstrip() for line in open('XOR-training.txt')]
        for line in alist:
            #print(line.split())
            trainingItem = line.split()
            a = int(trainingItem[0])
            b = int(trainingItem[1])
            inputs = [a, b]
            target = [int(trainingItem[2])]
            self.trainItem(inputs, target, learningRate)
