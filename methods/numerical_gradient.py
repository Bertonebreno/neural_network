from copy import deepcopy
from typing import List, Tuple

import numpy as np

from methods.feedforward import feedforward
from utils import cost


def numerical_gradient(
    theta: List[np.ndarray],
    theta0: List[np.ndarray],
    input: np.ndarray,
    expected_output: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    num_of_layers = len(theta)
    e = 1e-4

    gradient = [np.zeros(row.shape) if row is not None else None for row in theta]
    error = [np.zeros(row.shape) if row is not None else None for row in theta0]

    for layer in range(1, num_of_layers):
        for index, _ in np.ndenumerate(theta[layer]):
            theta_plus, theta_minus = deepcopy(theta), deepcopy(theta)

            theta_plus[layer][index] += e
            a_plus, _ = feedforward(theta_plus, theta0, input)

            theta_minus[layer][index] += -e
            a_minus, _ = feedforward(theta_minus, theta0, input)

            J_plus = cost(a_plus[-1], expected_output)
            J_minus = cost(a_minus[-1], expected_output)

            gradient[layer][index] += (J_plus - J_minus) / (2 * e)

    for layer in range(1, num_of_layers):
        for index, _ in np.ndenumerate(theta0[layer]):
            theta0_plus, theta0_minus = deepcopy(theta0), deepcopy(theta0)

            theta0_plus[layer][index] += e
            a_plus, _ = feedforward(theta, theta0_plus, input)

            theta0_minus[layer][index] += -e
            a_minus, _ = feedforward(theta, theta0_minus, input)

            J_plus = cost(a_plus[-1], expected_output)
            J_minus = cost(a_minus[-1], expected_output)

            error[layer][index] += (J_plus - J_minus) / (2 * e)

    return gradient, error
