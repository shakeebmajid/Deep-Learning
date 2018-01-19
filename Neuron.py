class Neuron:

    
    def ReLu(self, x):
        return max(0, x)

    def dReLu(self, x):
        if x > 0:
            return 1
        if x <= 0:
            return 0
