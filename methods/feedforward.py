from typing import List, Tuple

import numpy as np

from utils import sigmoid


def feedforward(
    theta: List[np.ndarray], theta0: List[np.ndarray], input: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    num_of_layers = len(theta)

    activated_neurons: List[np.ndarray] = [np.array([])] * num_of_layers
    neurons: List[np.ndarray] = [np.array([])] * num_of_layers

    activated_neurons[0] = input
    for layer in range(1, num_of_layers):
        neurons[layer] = (
            np.dot(theta[layer], activated_neurons[layer - 1]) + theta0[layer]
        )
        activated_neurons[layer] = sigmoid(neurons[layer])

    return activated_neurons, neurons
