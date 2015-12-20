import mnist_loader as ml
import network
import time
import numpy as np


def arctan_prime(z):
    return 1.0 / (1 + z**2)
    
def main():
    """
    Load data, set them into different sets:
    training data, validation data and test data
    """
    print "Loading MNIST data..."
    training_data, validation_data, test_data = ml.load_data_wrapper()

    """
    Initial the training set
    """
    net = network.Network([784,30,10])
    net.set_random_weights_and_biases()
    #net.set_transfer_function(np.arctan)
    #net.set_transfer_derivative(arctan_prime)
    
    epochs = 30
    mbs = 10 # mini-batch size
    eta = 0.5 # learning rate
    print "Training net: epochs={}, MBS={}, eta={}...".format(epochs, mbs, eta)

    """
    Trigger the training/test process
    Need to provice the epoch of training, mini-batch size and learning rate
    """
    start_time = time.time();
    net.SGD(training_data, epochs, mbs, eta)

    print "{} validation samples correctly classified".format(net.evaluate(validation_data))
    print "{} testing samples correctly classified".format(net.evaluate(test_data))
    print "------it takes %d seconds to complete the whole peocess-----" %(time.time() - start_time)

if __name__ == '__main__':
    main()
