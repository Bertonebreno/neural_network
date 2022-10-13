from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from methods.feedforward import feedforward
from methods.gradient_descent import gradient_descendent
from utils import generate_theta


def train_and_validate_model(
    num_of_neurons: List[int],
    training_images: List[np.ndarray],
    training_labels: List[np.ndarray],
    validation_images: List[np.ndarray],
    validation_labels: List[np.ndarray],
    num_of_iterations: int = 1000,
    batch_size: int = 100,
    regularization_constant: float = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:

    initial_theta, initial_theta0 = generate_theta(num_of_neurons)
    theta, theta0, cost_history = gradient_descendent(
        initial_theta,
        initial_theta0,
        training_images,
        training_labels,
        learning_rate=3,
        num_of_iterations=num_of_iterations,
        batch_size=batch_size,
        regularization_constant=regularization_constant,
        calc_numeric=False,
    )

    plt.plot(cost_history)
    plt.savefig(f"images/cost_history {str(num_of_neurons)}.png", dpi=300)

    trues = 0
    falses = 0
    for image, label in zip(validation_images, validation_labels):
        a, _ = feedforward(theta, theta0, image)
        if np.argmax(a[-1]) == np.argmax(label):
            trues += 1
        else:
            falses += 1

    return theta, theta0, trues / (trues + falses)
