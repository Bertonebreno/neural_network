from methods.feedforward import check_test_problem, feedforward, generate_test_problem
from methods.gradient_descendent import gradient_descendent
from utils import generate_theta, read_MNIST_data, plot_theta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_of_neurons = [2, 1]
num_of_layers = len(num_of_neurons)

(
    training_images,
    training_labels,
    validation_images,
    validation_labels,
    test_images,
    test_labels,
) = read_MNIST_data()

initial_theta, initial_theta0 = generate_theta(num_of_neurons)
input_data, expected_output = generate_test_problem(100)
theta, theta0, numeric_theta, numeric_theta0 = gradient_descendent(
    initial_theta,
    initial_theta0,
    input_data,
    expected_output,
    learning_rate=0.01,
    num_of_iterations=10000,
    calc_numeric=False,
)

handpicked_theta = [np.array([]), np.array([[10, -10]])]
handpicked_theta0 = [np.array([]), np.array([[0]])]

print(
    "Handpick test: ",
    check_test_problem(handpicked_theta, handpicked_theta0, number_of_tries=1000),
)
print("Backprop test: ", check_test_problem(theta, theta0, number_of_tries=1000))
print(
    "Numeric test: ",
    check_test_problem(numeric_theta, numeric_theta0, number_of_tries=1000),
)
