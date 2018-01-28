import numpy

class Neuron:
    def ReLu(self, x):
        return max(0, x)

    def dReLu(self, x):
        if x > 0:
            return 1
        if x <= 0:
            return 0

    def weightedSum(self, inputs, weights):
        weightedSum = 0

        for Input, weight in zip(inputs, weights):
            weightedSum += Input * weight

        return weightedSum

    def process(self, inputs, weights):
        #weightedSum = 0
        #i = 0
        #for Input, weight in zip(inputs, weights):
            #weightedSum += inputs[i] * weight
            #weightedSum += Input * weight
            #i += 1
        weightedSum = self.weightedSum(inputs, weights)
        return self.ReLu(weightedSum)

    def dProcess(self, inputs, weights):
        weightedSum = self.weightedSum(inputs, weights)
        return self.dReLu(weightedSum)

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def dSigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def output(self, inputs, weights):
        # weightedSum = 0
        # #i = 0
        # for Input, weight in zip(inputs, weights):
        #     #weightedSum += inputs[i] * weight
        #     weightedSum += Input * weight
        #     #i += 1

        weightedSum = self.weightedSum(inputs, weights)

        return self.sigmoid(weightedSum)

    def dOutput(self, inputs, weights):
        weightedSum = self.weightedSum(inputs, weights)

        return self.dSigmoid(weightedSum)

    def crossEntropy(self, inputs, weights, target):
        estimate = self.output(inputs, weights)

        crossEntropy = -(target * numpy.log(estimate + 0.000000000001) + (1 - target) * numpy.log(1 - estimate + 0.000000000001))

        return crossEntropy

    def dCost(self, inputs, weights, target):
        activation = self.output(inputs, weights)

        dCost = -(target / (activation + 0.000000000001)  - (1 - target) / (1 - activation + 0.000000000001))

        return dCost
