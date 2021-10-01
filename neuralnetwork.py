from __future__ import annotations

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
        """
        Make the layer.

        :param neuron_counts: How many neurons are in this layer.
        """
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
    def __init__(self, from_layer: int, from_neuron: int,
                 to_layer: int, to_neuron: int, weight: float):
        """
        Make a connection between neurons.

        :param from_layer: The layer the original neuron is on.
        :param from_neuron: The original neuron.
        :param to_layer: The layer the neuron to connect to is on.
        :param to_neuron: The neuron to connect to.
        """
        self.from_layer = from_layer
        self.from_neuron = from_neuron
        self.to_layer = to_layer
        self.to_neuron = to_neuron
        self.weight = weight
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


class NeuralNetwork:
    """
    A entire neural network composed of layers composed of neurons connected
    to each other.
    """
    def __init__(self, topology: list[int]):
        """
        Make a neural network.

        :param topology: A topology of the network, like [2, 3, 1] for
         3 layers with layer 1 having 2 neurons, layer 2 having 3 neurons,
         and layer 3 having 1 neuron.
        """
        self.layers = []
        self.bias = 0
        for neuron_count in topology:
            self.layers.append(Layer(neuron_count))
        self.layers[0].neurons.append(Neuron())
        self.layers[0].neurons[0].sum = 1
        self.fully_connect()

    def fully_connect(self):
        """
        Connect all the neurons to each other and initialize them all with
        random weights.
        """
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                try:
                    for next_neuron_index, next_neuron in enumerate(self.layers[layer_index + 1].neurons):
                        self.layers[layer_index].neurons[neuron_index].connections.append(
                            Connection(layer_index, neuron_index, layer_index + 1, next_neuron_index, randfloat(-1, 1))
                        )
                except IndexError:
                    pass
        self.layers[0].neurons[-1].connections.clear()
        self.layers[0].neurons[-1].sum = 1
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                self.layers[0].neurons[-1].connections.append(
                    Connection(layer_index, neuron_index, 0, self.bias, randfloat(-1, 1))
                )

    def reset_neurons(self):
        """
        Reset all the neurons.
        """
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                self.layers[layer_index].neurons[neuron_index].sum = 0
        self.layers[0].neurons[-1].sum = 1

    def sigmoid_layer(self, layer: int):
        """
        Sigmoid-ify layer.

        :param layer: The layer to run the sigmoid function on.
        """
        self.layers[layer].sigmoid()

    def step_layer(self, layer: int):
        """
        Step a layer forward.

        :param layer: The layer to step.
        """
        for neuron_index, neuron in enumerate(self.layers[layer].neurons):
            conn1 = self.layers[layer].neurons[neuron_index].connections
            for conn_index, conn in enumerate(conn1):
                layer2 = conn1[conn_index].to_layer
                self.layers[layer2].neurons[conn1[conn_index].to_neuron].sum += self.layers[layer].neurons[neuron_index].sum * conn1[conn_index].weight

    def feed_forward(self, insert_target: list[float]) -> list[float]:
        """
        Feed forward the entire neural network.

        :param insert_target: What to insert.
        :return: A list of floats.
        """
        for insert_index, insert in enumerate(insert_target):
            self.layers[0].neurons[insert_index].sum = insert
        for layer_index, layer in enumerate(self.layers):
            self.step_layer(layer_index)
            self.sigmoid_layer(layer_index)
        output = []
        for neuron_index, neuron in enumerate(self.layers[-1].neurons):
            output.append(self.layers[-1].neurons[neuron_index].sum)
        self.reset_neurons()
        return output

    def mutate(self, rate: float):
        """
        Mutate the entire network.

        :param rate: Mutation rate as a float.
        """
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                self.layers[layer_index].neurons[neuron_index].mutate(rate)

    def crossover(self, parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork:
        """
        Crossover self with other parent.

        :param parent1: A parent.
        :param parent2: The other parent.
        :return: Another neural network.
        """
        child = parent1
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                conns = parent2.layers[layer_index].neurons[neuron_index].connections
                for p2_n_idx, p2_n in enumerate(conns):
                    if randchance(0.5):
                        child.layers[layer_index].neurons[neuron_index].connections[p2_n_idx] = conns[p2_n_idx]
        return child
