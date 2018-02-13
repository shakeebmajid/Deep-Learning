import sys
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
        #self.dWeightMatrices = []
        #input size used for initializing first weight matrix
        inputSize = int(input('Input Size: '))
        #hidden layers initialized
        for i in range(depth):
            layerSize = int(input('Layer Size: '))
            weightMatrix = numpy.random.rand(layerSize, inputSize + 1)
            self.weightMatrices.append(weightMatrix)
            #self.dWeightMatrices.append(numpy.zeros(layerSize, inputSize + 1))
            layer = Layer(layerSize)
            self.network.append(layer)
            inputSize = layerSize

        #output layer which will likely have different activation functions
        layerSize = int(input('Output Layer Size: '))
        layer = Layer(layerSize)
        self.network.append(layer)

        weightMatrix = numpy.random.rand(layerSize, inputSize + 1)
        self.weightMatrices.append(weightMatrix)
        #self.dWeightMatrices.append(numpy.zeros(layerSize, inputSize + 1))

    def feedForward(self, inputs, targets):
        #passing outputs from one layer to the next
        #print("weight matrices: ", self.weightMatrices)
        self.activations = []
        self.dOutputs = []
        self.dCosts = []
        self.inputs = inputs
        for i, weightMatrix in zip(range(self.depth), self.weightMatrices):
            layer = self.network[i]
            #bias of positive 1
            inputs.append(1)
            inputs = layer.feedForward(inputs, weightMatrix)
            self.activations.append(inputs)
            self.dOutputs.append(layer.dReLus(inputs, weightMatrix))

        #bias for output layer
        inputs.append(1)

        #output of output layer
        outputLayer = self.network[self.depth]
        outputs = outputLayer.outputs(inputs, self.weightMatrices[self.depth])
        self.activations.append(outputs)
        self.dOutputs.append(outputLayer.dOutputs(inputs, self.weightMatrices[self.depth]))
        self.dCosts = outputLayer.dCosts(inputs, self.weightMatrices[self.depth], targets)

        #calcuate cost of training item
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
            #print "stored matrix:", self.weightMatrices[l-1]
            previousWeightMatrix = numpy.array(self.weightMatrices[l + 1][:, :-1])
            #print "matrix minus column:", previousWeightMatrix
            previousWeightTranspose = previousWeightMatrix.transpose()
            previousDelta = self.delta(l + 1)
            dReLus = numpy.array(self.dOutputs[l])
            #print("dRelus: ", dReLus)
            #print("transpose: ", previousWeightTranspose)
            #print("previous delta: ", previousDelta)
            #print "previous weight * previousDelta:", previousWeightTranspose.dot(previousDelta)
            return numpy.multiply(previousWeightTranspose.dot(previousDelta), dReLus)


    def dWeights(self, l):
        if l == 0:
            activations = numpy.array(self.inputs)
        else:
            activations = numpy.array(self.activations[l - 1])

        #transposeDeltas = numpy.array(self.delta(l)).transpose()
        deltas = numpy.array(self.delta(l))

        #bias
        #activations = numpy.append(activations, [1])
        #print "activations:", activations, "deltas:", deltas

        #dWeights = transposeDeltas.dot(activations)
        dWeights = numpy.outer([deltas], [activations])
        return dWeights


    def backProp(self):
        network = self.network
        dWeightMatrices = (0 * numpy.array(self.weightMatrices)).tolist()
        for i, layer in reversed(list(enumerate(network))):
            # print i, "weights: ", self.weightMatrices[i], "dCost / dWeights: ", self.dWeights(i)
            #print("pre: ", self.weightMatrices[i])
            #print("pre: ", self.weightMatrices)
            dWeightMatrices[i] = self.dWeights(i)
            #print("change: ", self.weightMatrices[i] - learningRate * self.dWeights(i))
            #print("post: ", self.weightMatrices[i])
            #print("post: ", self.weightMatrices)

            #print self.dWeights(i)

        return dWeightMatrices

    def gradientDescent(self, learningRate, dWeightsAverage):
        self.weightMatrices = (numpy.array(self.weightMatrices) - (learningRate * numpy.array(dWeightsAverage)).tolist()).tolist()


    def trainItem(self, inputs, targets, learningRate):
        self.feedForward(inputs, targets)
        self.backProp(learningRate)

    #used for early stopping and saves the lowest cost encountered during last training run
    def saveLowestState(self, cost):
        if cost < self.lowestCost:
            self.lowestWeights = self.weightMatrices
            self.lowestCost = cost

    #used to set network to state with lowest cost on last training run
    def setToLowestState(self):
        self.weightMatrices = self.lowestWeights
        self.cost = self.lowestCost


    def train(self, trainingSize, learningRate, batchSize):
        #write
        f = open("XOR-training.txt", "w+")
        for i in range(trainingSize):
            a = bool(random.getrandbits(1))
            b = bool(random.getrandbits(1))
            f.write (str(int(a)) + " " + str(int(b)) + " " + str(int(a ^ b)) + "\n")
            #f.write("This is line %d\r\n" % (i+1))

        f.close()


        alist = [line.rstrip() for line in open('XOR-training.txt')]
        batchNum = 1
        accumulatedWeights = (0 * numpy.array(self.weightMatrices)).tolist()
        totalCost = 0
        i = 1
        self.lowestCost = sys.maxint
        for line in alist:
            #print(line.split())
            trainingItem = line.split()
            a = int(trainingItem[0])
            b = int(trainingItem[1])
            inputs = [a, b]
            target = [int(trainingItem[2])]
            self.feedForward(inputs, target)
            accumulatedWeights = (numpy.array(accumulatedWeights) + numpy.array(self.backProp())).tolist()
            #self.trainItem(inputs, target, learningRate)
            totalCost += self.cost
            if i == batchSize:
                dWeightsAverage = (numpy.array(accumulatedWeights) / batchSize).tolist()
                averageCost = totalCost / batchSize
                print "total cost:", totalCost, "average cost:", averageCost
                self.saveLowestState(averageCost)
                if averageCost == 0:
                    break

                self.gradientDescent(learningRate, dWeightsAverage)
                i = 0
                totalCost = 0
                accumulatedWeights = (0 * numpy.array(self.weightMatrices)).tolist()
                print "Batch #", batchNum
                batchNum += 1
            i += 1
