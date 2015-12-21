import mnist_loader
import network2
import time
import numpy as np

def arctan(z):
    """
    The original arctan function is in the range (-PI/2, +PI/2).
    This variation transforms the function to be in the range (0, 1).
    """
    return np.arctan(z) / np.pi + 0.5

def arctan_prime(z):
    return 1.0 / (1 + z**2) / np.pi

def main():
	print "Using network 2..."

	"""
	Loading data
	"""
	print "Loading data..."
	training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

	"""
	initialize the training network with network2
	"""
	hidden_size = 50
	net = network2.Network([784, hidden_size, 10])
	net.large_weight_initializer()

	"""
	set transfer function to arctan
	with corresponding derivative function
	"""
	net.set_transfer_function(arctan)
	net.set_transfer_derivative(arctan_prime)
	print "Using arctan as sigmoid..."

	epochs = 50
	mini_batch_size = 20
	learning_rate = 0.5

	"""
	Trigger the training process
	"""
	print "Training net: epochs={}, batch_size={}, learning_rate={}, hidden={}...".format(
		epochs, mini_batch_size, learning_rate, hidden_size)

	start_time = time.time()
	net.SGD(training_data, epochs, mini_batch_size, learning_rate, evaluation_data=test_data, monitor_evaluation_accuracy=True)
	print "It takes %d seconds to complete the whole process" %(time.time() - start_time)
if __name__ == '__main__':
    main()

