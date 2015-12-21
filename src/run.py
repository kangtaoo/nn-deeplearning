import mnist_loader as ml
import network
import time
import numpy as np

def arctan(z):
    return np.arctan(z) / np.pi + 0.5

def arctan_prime(z):
    return 1.0 / (1 + z**2) / np.pi

def tanh(z):
    return np.tanh(z) / 2 + 0.5

def tanh_prime(z):
    return (1 - np.tanh(z)**2) / 2
    
def main():
    print "Loading MNIST data..."
    training_data, validation_data, test_data = ml.load_data_wrapper()

    # Initialize the network
    net = network.Network([784,30,10])
    net.set_random_weights_and_biases()
    net.set_transfer_function(arctan)
    net.set_transfer_derivative(arctan_prime)
    
    epochs = 30
    mbs = 10 # mini-batch size
    eta = 0.75 # learning rate
    print "Training net: epochs={}, MBS={}, eta={}...".format(epochs, mbs, eta)

    start_time = time.time()
    net.SGD(training_data, epochs, mbs, eta, validation_data)

    print "{} testing samples correctly classified".format(net.evaluate(test_data))
    print "------it takes %d seconds to complete the whole process-----" %(time.time() - start_time)

if __name__ == '__main__':
    main()
