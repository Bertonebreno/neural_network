from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from methods.backpropagation import backpropagation
from methods.feedforward import feedforward
from methods.numerical_gradient import numerical_gradient
from utils import cost, split_data


def gradient_descendent(
    initial_theta: List[np.ndarray],
    initial_theta0: List[np.ndarray],
    full_input_data: List[np.ndarray],
    full_expected_output: List[np.ndarray],
    learning_rate: float,
    num_of_iterations: int,
    batch_size: Optional[int] = None,
    calc_numeric: bool = False,
) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[float]
]:
    theta: List[np.ndarray] = deepcopy(initial_theta)
    theta0: List[np.ndarray] = deepcopy(initial_theta0)

    num_of_layers = len(theta)
    num_of_examples = batch_size or len(full_input_data)

    cost_history = []
    for i in range(num_of_iterations):
        input_data, expected_output = split_data(
            full_input_data, full_expected_output, batch_size
        )

        if i % int(num_of_iterations / 10) == 0:
            print(100 * "\n")
            print(f"Progress: {i}/{num_of_iterations}")

        gradient, error = get_gradient_shaped_array(theta, theta0)
        numeric_gradient, numeric_error = get_gradient_shaped_array(theta, theta0)

        total_cost = 0
        for input_row, expected_output_row in zip(input_data, expected_output):
            activated_neurons, neurons = feedforward(theta, theta0, input_row)
            total_cost += cost(activated_neurons[-1], expected_output_row) / num_of_examples

            gradient_row, error_row = backpropagation(
                theta, theta0, neurons, activated_neurons, expected_output_row
            )
            if calc_numeric:
                numeric_gradient_row, numeric_error_row = numerical_gradient(
                    theta, theta0, input_row, expected_output_row
                )

            for layer in range(1, num_of_layers):
                gradient[layer] -= gradient_row[layer] / num_of_examples
                error[layer] -= error_row[layer] / num_of_examples

                if calc_numeric:
                    numeric_gradient[layer] -= (
                        numeric_gradient_row[layer] / num_of_examples
                    )
                    numeric_error[layer] -= numeric_error_row[layer] / num_of_examples

        for layer in range(1, num_of_layers):
            theta[layer] += learning_rate * gradient[layer]
            theta0[layer] += learning_rate * error[layer]

        cost_history.append(total_cost)


    return theta, theta0, cost_history


def get_gradient_shaped_array(
    theta: List[np.ndarray], theta0: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    gradient = [np.zeros(row.shape) if row is not None else None for row in theta]
    error = [np.zeros(row.shape) if row is not None else None for row in theta0]

    return gradient, error
