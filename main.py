from config.log_config import logger as main_logger
from methods.train_and_validate_model import train_and_validate_model
from utils import plot_theta, read_MNIST_data

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


theta, theta0, performance = train_and_validate_model(
    num_of_neurons,
    training_images,
    training_labels,
    validation_images,
    validation_labels,
    num_of_iterations=1000,
    batch_size=100,
    regularization_constant=0,
)
logger.info(
    f" Network {num_of_neurons} trained on {len(training_images)} examples and had {100*performance:.2f}% of performance in validation set"
)

plot_theta(theta, num_rows=4, num_cols=8, description=str(num_of_neurons))
