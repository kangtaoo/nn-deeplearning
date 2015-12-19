import mnist_loader as ml
import network
import numpy as np

def arctan_prime(z):
    return 1.0 / (1 + z**2)
    
def main():
    print "Loading MNIST data..."
    training_data, validation_data, test_data = ml.load_data_wrapper()

    net = network.Network([784,30,10])
    net.set_random_weights_and_biases()
    #net.set_transfer_function(np.arctan)
    #net.set_transfer_derivative(arctan_prime)
    
    epochs = 30
    mbs = 10 # mini-batch size
    eta = 3.0 # learning rate
    print "Training net: epochs={}, MBS={}, eta={}...".format(epochs, mbs, eta)
    net.SGD(training_data, epochs, mbs, eta)

    print "{} validation samples correctly classified".format(net.evaluate(validation_data))
    print "{} testing samples correctly classified".format(net.evaluate(test_data))

if __name__ == '__main__':
    main()
