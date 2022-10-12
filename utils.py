from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_theta(
    num_of_neurons: List[int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    num_of_layers = len(num_of_neurons)

    theta: List[np.ndarray] = [np.array([])] * num_of_layers  # Pesos
    theta0: List[np.ndarray] = [np.array([])] * num_of_layers  # Biases

    for layer in range(1, num_of_layers):
        epi = (6**(1/2))/(num_of_neurons[layer] + num_of_neurons[layer - 1])**(1/2)

        theta[layer] = np.random.rand(num_of_neurons[layer], num_of_neurons[layer - 1]) * epi
        theta0[layer] = np.random.rand(num_of_neurons[layer], 1) * epi

    return theta, theta0


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def d_sigmoid(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))


def cost(output: np.ndarray, expected_output: np.ndarray) -> float:
    return np.sum((output - expected_output) ** 2 / 2)


def plot_theta(theta: List[np.ndarray]) -> None:
    fig, axis = plt.subplots(4, 8, figsize=(8, 8))
    image_number = 0
    for i in range(4):
        for j in range(8):
            theta_Li = theta[1][image_number]
            axis[i, j].imshow(
                (theta_Li - np.min(theta_Li) / np.ptp(theta_Li)).reshape(
                    20, 20, order="F"
                ),
                cmap="hot",
            )
            axis[i, j].axis("off")
            image_number += 1
    plt.show()


def read_MNIST_data() -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
]:
    images = pd.read_csv(
        "data/imageMNIST.csv", delimiter="|", quotechar='"', header=None
    ).values.tolist()
    labels = pd.read_csv(
        "data/labelMNIST.csv", delimiter=",", quotechar='"', header=None
    ).values.tolist()

    training_images = images[:3000]
    training_labels = labels[:3000]

    validation_images = images[3000:4001]
    validation_labels = labels[3000:4001]

    test_images = images[4000 : 5000 - 1]
    test_labels = labels[4000 : 5000 - 1]

    return (
        training_images,
        training_labels,
        validation_images,
        validation_labels,
        test_images,
        test_labels,
    )
