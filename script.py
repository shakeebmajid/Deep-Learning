import Neuron
import Layer
import FeedForwardNetwork
import random

from Neuron import *
from Layer import *
from FeedForwardNetwork import *

inputs = [5, 5, 5]

weights = [5, 5 , 50]

n = Neuron()

l = Layer(5)

network = FeedForwardNetwork(1)

print "inputs: [1, 1], target: 0, output:", network.feedForward([1, 1], [0])
print "Cost:", network.cost
print "inputs: [0, 0], target: 0, output:", network.feedForward([0, 0], [0])
print "Cost:", network.cost
print "inputs: [1, 0], target: 1, output:", network.feedForward([1, 0], [1])
print "Cost:", network.cost
print "inputs: [0, 1], target: 1, output:", network.feedForward([0, 1], [1])
print "Cost:", network.cost

print "weight matrices:", network.weightMatrices

trainingSize = int(input('Training Size: '))
learningRate = 0.05
network.train(trainingSize, learningRate, 5)

print "inputs: [1, 1], target: 0, output:", network.feedForward([1, 1], [0])
print "Cost:", network.cost
print "inputs: [0, 0], target: 0, output:", network.feedForward([0, 0], [0])
print "Cost:", network.cost
print "inputs: [1, 0], target: 1, output:", network.feedForward([1, 0], [1])
print "Cost:", network.cost
print "inputs: [0, 1], target: 1, output:", network.feedForward([0, 1], [1])
print "Cost:", network.cost

print "weight matrices:", network.weightMatrices


# network = FeedForwardNetwork(1)
#
# #print(network.feedForward([5, 5], [0.5, 0.5]))
#
# #print(network.dWeights(1))
#
#
# #write
# f = open("XOR-training.txt", "w+")
# for i in range(10000):
#     a = bool(random.getrandbits(1))
#     b = bool(random.getrandbits(1))
#     f.write (str(int(a)) + " " + str(int(b)) + " " + str(int(a ^ b)) + "\n")
#     #f.write("This is line %d\r\n" % (i+1))
#
# f.close()
#
# #read
# alist = [line.rstrip() for line in open('XOR-training.txt')]
# for line in alist:
#     #print(line.split())
#     trainingItem = line.split()
#     a = int(trainingItem[0])
#     b = int(trainingItem[1])
#     inputs = [a, b]
#     target = [int(trainingItem[2])]
#     learningRate = 0.05
#     network.trainItem(inputs, target, learningRate)
#
#
# print network.feedForward([1, 1], [0])
# print network.feedForward([0, 0], [0])
# print network.feedForward([1, 0], [1])
# print network.feedForward([0, 1], [1])

# f.close()
#
# w = network.weightMatrices
# w0 = w[0]
# w1 = w[1]
# w2 = w[2]
