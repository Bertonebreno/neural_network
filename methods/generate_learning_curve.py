from typing import List

import matplotlib.pyplot as plt
import numpy as np

from config.log_config import logger as main_logger
from methods.feedforward import feedforward
from methods.gradient_descent import gradient_descendent
from utils import cost, generate_theta

logger = main_logger.getChild(__name__)


def generate_learning_curve(
    num_of_neurons: List[int],
    training_images: List[np.ndarray],
    training_labels: List[np.ndarray],
    validation_images: List[np.ndarray],
    validation_labels: List[np.ndarray],
    num_of_iterations: int = 1000,
    batch_size: int = 150,
    regularization_constant: float = 0,
) -> None:
    plt.clf()
    
    training_cost_history: List[float] = []
    validation_cost_history: List[float] = []
    number_of_training_examples: List[int] = []

    for i in range(150, len(training_images), 150):
        logger.info(f" Calculating cost with {i} training examples")
        number_of_training_examples.append(i)
        initial_theta, initial_theta0 = generate_theta(num_of_neurons)
        theta, theta0, cost_history = gradient_descendent(
            initial_theta,
            initial_theta0,
            training_images[:i],
            training_labels[:i],
            learning_rate=3,
            num_of_iterations=num_of_iterations,
            batch_size=min(batch_size, i),
            regularization_constant=regularization_constant,
            calc_numeric=False,
        )
        training_cost = cost_history[-1]
        validation_cost = 0.0
        for image, label in zip(validation_images, validation_labels):
            a, _ = feedforward(theta, theta0, image)
            validation_cost += cost(a[-1], label) / len(validation_images)

        training_cost_history.append(training_cost)
        validation_cost_history.append(validation_cost)

    plt.plot(
        number_of_training_examples, training_cost_history, "red", label="Training cost"
    )
    plt.plot(
        number_of_training_examples,
        validation_cost_history,
        "green",
        label="Validation cost",
    )
    plt.legend()
    plt.savefig(f"images/learning_curve {num_of_neurons}.png", dpi=300)
