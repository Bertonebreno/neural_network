from typing import List, Tuple

import numpy as np

from utils import d_sigmoid


def backpropagation(
    theta: List[np.ndarray],
    theta0: List[np.ndarray],
    neurons: List[np.ndarray],
    activated_neurons: List[np.ndarray],
    expected_output: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    num_of_layers = len(theta)

    gradient = [np.zeros(row.shape) if row is not None else None for row in theta]
    error = [np.zeros(row.shape) if row is not None else None for row in theta0]

    error[num_of_layers - 1] += (
        activated_neurons[num_of_layers - 1] - expected_output
    ) * d_sigmoid(neurons[num_of_layers - 1])
    for layer in range(num_of_layers - 2, 0, -1):
        error[layer] += np.dot(theta[layer + 1].T, error[layer + 1]) * d_sigmoid(
            neurons[layer]
        )

    for layer in range(1, num_of_layers):
        gradient[layer] += np.dot(error[layer], activated_neurons[layer - 1].T)

    return gradient, error
