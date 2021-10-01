from math import e
from random import random, uniform as randfloat


def sigmoid(x: float) -> float:
    """
    Sigmoid function.

    :param x: The value to sigmoid-ify.
    :return: A float.
    """
    if x <= 0:
        return 0
    return 1 / (1 + e ** (-4.9 * x))


def randchance(chance: float) -> bool:
    """
    Return a bool depending on the chance.

    :param chance: A float from 0 to 1.
    :return: A bool.
    """
    return random() <= chance


class Neuron:
    """
    A single neuron.
    """
    def __init__(self):
        self.connections = []
        self.sum = 0

    def mutate(self, rate: float):
        """
        Mutate some of the connections depending on rate.

        :param rate: A float from 0 to 1 depending on how often to mutate.
        """
        for connection in self.connections:
            if randchance(rate):
                connection.mutateWeight()


class Layer:
    """
    A layer of neurons.
    """
    def __init__(self, neuron_counts: int):
        self.neurons = []
        for _ in range(neuron_counts):
            self.neurons.append(Neuron())

    def sigmoid(self):
        """
        Run the sigmoid function on all the neurons.
        """
        for neuron in self.neurons:
            neuron.sum = sigmoid(neuron.sum)


class Connection:
    """
    A connection from a neuron in a layer to another neuron in another layer.
    """
    def __init__(self, from_layer: Layer, from_neuron: Neuron,
                 to_layer: Layer, to_neuron: Neuron):
        self.from_layer = from_layer
        self.from_neuron = from_neuron
        self.to_layer = to_layer
        self.to_neuron = to_neuron
        self.weight = 0
        self.min_max = 1

    def mutateWeight(self):
        """
        Mutate the weights.
        """
        if randchance(0.1):
            self.weight = randfloat(-self.min_max, self.min_max)
        else:
            self.weight = randfloat(-0.02, 0.02)
        if self.weight > self.min_max:
            self.weight = self.min_max
        if self.weight < -self.min_max:
            self.weight = -self.min_max
