from methods.feedforward import check_test_problem, feedforward, generate_test_problem
from methods.gradient_descent import gradient_descendent
from utils import generate_theta, read_MNIST_data, plot_theta
import matplotlib.pyplot as plt
import numpy as np

from config.log_config import logger as main_logger

logger = main_logger.getChild(__name__)

num_of_neurons = [400, 32, 32, 10]
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

theta, theta0, cost_history= gradient_descendent(
    initial_theta,
    initial_theta0,
    training_images,
    training_labels,
    learning_rate=0.01,
    num_of_iterations=1000,
    batch_size=100,
    calc_numeric=False,
)

print(len(cost_history))

plt.plot(cost_history)
plt.show()

trues = 0
falses = 0
for i in range(100):
    a, z = feedforward(theta, theta0, validation_images[i])
