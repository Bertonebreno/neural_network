from typing import List, Tuple

import numpy as np


def generate_theta(
    num_of_neurons: List[int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    num_of_layers = len(num_of_neurons)

    theta: List[np.ndarray] = [np.array([])] * num_of_layers  # Pesos
    theta0: List[np.ndarray] = [np.array([])] * num_of_layers  # Biases

    for layer in range(1, num_of_layers):
        theta[layer] = np.random.rand(num_of_neurons[layer], num_of_neurons[layer - 1])
        theta0[layer] = np.random.rand(num_of_neurons[layer], 1)

    return theta, theta0


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))


def cost(output: np.ndarray, expected_output: np.ndarray) -> float:
    return np.sum((output - expected_output) ** 2 / 2)
