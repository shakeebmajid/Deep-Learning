import Neuron
import Numpy
class Layer:
    def __init__(self, neurons):
        self.neurons = neurons


    def feedForward(self, weightMatrix, inputs):
        i = 0
        outputs = []
        for neuron in neurons:
            weights = weightMatrix[i]
            outputs.append(neuron.process(inputs, weights))
            i += 1

        return outputs

    
