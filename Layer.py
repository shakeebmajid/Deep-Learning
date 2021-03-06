import numpy
import Neuron
from Neuron import *

class Layer:
    def __init__(self, width):
        self.neurons = []
        for i in range(width):
            self.neurons.append(Neuron())


    def feedForward(self, inputs, weightMatrix):
        #i = 0
        outputs = []
        for neuron, weights in zip(self.neurons, weightMatrix):
            #weights = weightMatrix[i]
            outputs.append(neuron.process(inputs, weights))
            #i += 1
        return outputs

    def outputs(self, inputs, weightMatrix):
        outputs = []
        for neuron, weights in zip(self.neurons, weightMatrix):
            #weights = weightMatrix[i]
            outputs.append(neuron.output(inputs, weights))
            #i += 1

        return outputs

    def dOutputs(self, inputs, weightMatrix):
        dOutputs = []
        for neuron, weights in zip(self.neurons, weightMatrix):
            dOutputs.append(neuron.dOutput(inputs, weights))

        return dOutputs

    def dReLus(self, inputs, weightMatrix):
        dReLus = []
        for neuron, weights in zip(self.neurons, weightMatrix):
            dReLus.append(neuron.dProcess(inputs, weights))

        return dReLus

    def costs(self, inputs, weightMatrix, targets):
        costs = []
        for neuron, weights, target in zip(self.neurons, weightMatrix, targets):
            costs.append(neuron.crossEntropy(inputs, weights, target))

        return costs

    def dCosts(self, inputs, weightMatrix, targets):
        dCosts = []
        for neuron, weights, target in zip(self.neurons, weightMatrix, targets):
            dCosts.append(neuron.dCost(inputs, weights, target))

        return dCosts
