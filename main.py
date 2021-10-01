from random import randint
from neuralnetwork import NeuralNetwork


topology = [2, 2, 2, 1]

networks = []
fitnesses = []

for _ in range(200):
    networks.append(NeuralNetwork(topology))
    fitnesses.append(0)

outputs = []
best_ever = NeuralNetwork(topology)
best_ever_fitness = 0

all_inserts = [[0, 1], [0, 0], [1, 0], [1, 1]]

for round in range(2000):
    inputs = [randint(0, 1), randint(0, 1)]
    for network_index, network in enumerate(networks):
        for insert_index, insert in enumerate(all_inserts):
            inputs = [all_inserts[insert_index][0], all_inserts[insert_index][1]]
            output = network.feed_forward(inputs)
            expected = 0
            if inputs[0] == inputs[1] and output[0] < 0.5:
                fitnesses[network_index] += 1 - output[0]
            elif output[0] > 0.5 and inputs[0] != inputs[1]:
                fitnesses[network_index] += output[0] - 0.5
            else:
                fitnesses[network_index] -= 1
            error = abs(expected - output[0])
    best = 0
    highest = 0
    elitists = [0, 0]
    for network_index, network in enumerate(networks):
        if highest < fitnesses[network_index]:
            elitists[0] = elitists[1]
            elitists[1] = network_index
            highest = fitnesses[network_index]
            best = network_index
        fitnesses[network_index] = 0
    best_network = networks[elitists[0]]
    if round % 100 == 0:
        print(f"Round {round} highest fitness: {highest}")
    if highest > best_ever_fitness:
        best_ever_fitness = highest
        best_ever = best_network
        print(f"New highest fitness: {highest}")
    for network_index, network in enumerate(networks):
        networks[network_index] = networks[network_index].crossover(networks[elitists[0]], networks[elitists[1]])
        networks[network_index].mutate(0.2)

for _ in range(30):
    test_input = [randint(0, 1), randint(0, 1)]
    output = best_ever.feed_forward(test_input)
    print(f"Test case: {test_input}")
    print(f"Output: {output}")
