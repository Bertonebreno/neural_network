from typing import List, Tuple

import numpy as np

from utils import sigmoid


def feedforward(
    theta: List[np.ndarray], theta0: List[np.ndarray], input: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    num_of_layers = len(theta)

    activated_neurons = [np.array([])] * num_of_layers
    neurons = [np.array([])] * num_of_layers

    activated_neurons[0] = input
    for layer in range(1, num_of_layers):
        neurons[layer] = (
            np.dot(theta[layer], activated_neurons[layer - 1]) + theta0[layer]
        )
        activated_neurons[layer] = sigmoid(neurons[layer])

    return activated_neurons, neurons


def generate_test_problem(num_of_examples: int) -> Tuple[np.ndarray, np.ndarray]:
    input = np.array([np.random.random((2, 1)) for i in range(num_of_examples)])
    expected_output = np.array([1 if row[0] > row[1] else 0 for row in input])

    return input, expected_output


def check_test_problem(
    theta: List[np.ndarray], theta0: List[np.ndarray], number_of_tries: int
) -> Tuple[int, int]:
    trues = 0
    falses = 0
    for i in range(number_of_tries):
        input, expected_output = generate_test_problem(1)

        activated_neurons, _ = feedforward(theta, theta0, input)

        answer = True if activated_neurons[-1] > 0.5 else False
        if answer == expected_output:
            trues += 1
        else:
            falses += 1
    return trues, falses
