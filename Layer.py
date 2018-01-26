import numpy
class Layer:
    def __init__(self, layerSize):
        self.neurons = []
        for i in range(layerSize):
            neuron = Neuron()
            neurons.append(neuron)


    def feedForward(self, inputs, weightMatrix):
        #i = 0
        outputs = []
        for neuron, weights in zip(self.neurons, weightMatrix):
            #weights = weightMatrix[i]
            outputs.append(neuron.process(inputs, weights))
            #i += 1

        return outputs

    def output(self, inputs, weightMatrix):
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
        return cost
