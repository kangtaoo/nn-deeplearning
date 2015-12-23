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
    print 'Loading MNIST data...'
    training_data, validation_data, test_data = ml.load_data_wrapper()
    training_data2 = [(x,np.argmax(y)) for x,y in training_data]

    hidden = 50
    func_name = 'arctan' # sigmoid, arctan, or tanh

    # Initialize the network
    net = network.Network([784,hidden,10])
    net.set_random_weights_and_biases()

    if func_name == 'sigmoid':
        net.set_transfer_function(network.sigmoid)
        net.set_transfer_derivative(network.sigmoid_prime)
    elif func_name == 'arctan':
        net.set_transfer_function(arctan)
        net.set_transfer_derivative(arctan_prime)
    elif func_name == 'tanh':
        net.set_transfer_function(tanh)
        net.set_transfer_derivative(tanh_prime)
    
    epochs = 50
    mbs = 1 # mini-batch size
    eta = 0.9 # learning rate
    print 'Training net: epochs={}, MBS={}, eta={}, hidden={}, function={}...'.format(epochs, mbs, eta, hidden, func_name)

    start_time = time.time()
    net.SGD(training_data, epochs, mbs, eta, validation_data)
    duration = time.time() - start_time

    print '{} / {} training samples correctly classified'.format(net.evaluate(training_data2), len(training_data2))
    print '{} / {} testing samples correctly classified'.format(net.evaluate(test_data), len(test_data))
    print 'Process duration: {} seconds'.format(duration)

if __name__ == '__main__':
    main()
