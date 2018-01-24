import numpy
class Layer:
    def __init__(self, layerSize):
        self.neurons = []
        for i in range(layerSize):
            neuron = Neuron()
            neurons.append(neuron)


    def feedForward(self, inputs, weightMatrix):
        i = 0
        outputs = []
        for neuron in self.neurons:
            weights = weightMatrix[i]
            outputs.append(neuron.process(inputs, weights))
            i += 1

        return outputs

    def layerCost(self, inputs, weightMatrix, targets):
        cost = 0

        for neuron, weights, target in zip(self.neurons, weightMatrix, targets):
            cost += neuron.crossEntropy(inputs, weights, target)

        return cost
