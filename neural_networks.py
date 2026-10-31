import numpy as np

class Neuron:
    def __init__(self,weights,bias):
        self.weights = np.array(weights)
        self.bias = bias

    def heaviside(self,inputs):
        z = np.dot(self.weights,inputs) + self.bias

        return 1 if z>0 else 0

    def sigmoid(self,inputs):
        z = np.dot(self.weights,inputs) + self.bias

        return 1 / (1+np.exp(-z))

neuron21 = Neuron(
    weights = [0.5,-1],
    bias = 1
)

neuron22 = Neuron(
    weights = [1,-0.5],
    bias = 4
)

neuron23 = Neuron(
    weights = [0.5,2],
    bias = 3
)

neuron31 = Neuron(
    weights = [1,-0.5,2],
    bias = 3
)

neuron32 = Neuron(
    weights = [0.5,2,-1.5],
    bias = 2
)

x1=[2,3]

x2=np.array([neuron21.sigmoid(x1),neuron22.sigmoid(x1),neuron23.sigmoid(x1)])

x3=np.array([neuron31.sigmoid(x2),neuron32.sigmoid(x2)])

print(x2)
print(x3)