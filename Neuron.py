class Neuron:
    def ReLu(self, x):
        return max(0, x)

    def dReLu(self, x):
        if x > 0:
            return 1
        if x <= 0:
            return 0

    def process(self, inputs, weights):
        weightedSum = 0
        i = 0
        for weight in weights:
            weightedSum += inputs[i] * weight
            i += 1

        return self.ReLu(weightedSum)
