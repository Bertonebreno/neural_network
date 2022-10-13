from config.log_config import logger as main_logger
from methods.train_and_validate_model import train_and_validate_model
from utils import NeuralNetwork, plot_theta, read_MNIST_data

logger = main_logger.getChild(__name__)

possible_num_of_neurons = [
    [400, 10, 10],
    [400, 25, 10],
    [400, 32, 32, 10],
]
possible_regularization_constants = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10.]

(
    training_images,
    training_labels,
    validation_images,
    validation_labels,
    test_images,
    test_labels,
) = read_MNIST_data()


# Empty network to start the loop
best_network = NeuralNetwork(
    num_of_neurons=[],
    theta=[],
    theta0=[],
    regularization_constant=0,
    performance=0,
    training_time=0,
    mean_time_in_feedforward=0,
)
for num_of_neurons in possible_num_of_neurons:
    for regularization_constant in possible_regularization_constants:
        network = train_and_validate_model(
            num_of_neurons,
            training_images,
            training_labels,
            validation_images,
            validation_labels,
            num_of_iterations=1000,
            batch_size=100,
            regularization_constant=regularization_constant,
        )
        logger.info(
            f" Network {num_of_neurons} trained on {len(training_images)} examples and had {100*network.performance:.2f}% of performance in validation set"
        )
        if network.performance > best_network.performance:
            best_network = network

logger.info(
    f"The best network is {best_network.num_of_neurons} with λ={best_network.regularization_constant}, which had {100*best_network.performance:.2f}% of performance in validation set"
)
logger.info(f"Its training time is: {best_network.training_time:.2f} seconds")
logger.info(
    f"The mean time elapsed in feedforward for this network is: {1000*best_network.mean_time_in_feedforward:.5f} milliseconds"
)

aux = {10: [2, 5], 25: [5, 5], 32: [4, 8]}
plot_theta(
    best_network.theta,
    num_rows=aux[num_of_neurons[1]][0],
    num_cols=aux[num_of_neurons[1]][1],
    description=str(num_of_neurons),
)
