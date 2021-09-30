from math import e


def sigmoid(x: float) -> float:
    """
    Sigmoid function from
    https://github.com/Bobingstern/CPP_NeuralNetwork/blob/main/main.cpp#L56.

    :param x: The value to sigmoid-ify.
    :return: A float.
    """
    if x <= 0:
        return 0
    return 1 / (1 + e ** (-4.9 * x))
