import mnist_loader as ml
import network
import time
import numpy as np
import sys

def arctan(z):
    return np.arctan(z) / np.pi + 0.5

def arctan_prime(z):
    return 1.0 / (1 + z**2) / np.pi

def tanh(z):
    return np.tanh(z) / 2 + 0.5

def tanh_prime(z):
    return (1 - np.tanh(z)**2) / 2

def show_menu():
    print 'q: Quit        h: Show this menu        r: Reset NN'
    print 'l: Set learning rate         m: Set mini-batch size'
    print 'e: Run [more] epochs on NN'
    print 'n: Test against training data'
    print 't: Test against testing data'

def prompt_for_action():
    input = raw_input('Enter a character [e]: ').lower()
    return 'e' if len(input)==0 else input[0]

def prompt_for_net_params():
    nh = raw_input('Enter # of hidden neurons [30]:')
    if nh == '': nh = 30
    
    net = network.Network([784,int(nh),10])
    
    rwb = raw_input('Randomize weights and biases? [y]/n: ').lower()
    if rwb == 'y' or rwb == '':
        net.set_random_weights_and_biases()
    
    tf = raw_input('Which transfer function (sigmoid/arctan/tanh)? s/[a]/h: ').lower()
    if tf == 's':
        net.set_transfer_function(network.sigmoid)
        net.set_transfer_derivative(network.sigmoid_prime)
    elif tf == 'a' or tf == '':
        net.set_transfer_function(arctan)
        net.set_transfer_derivative(arctan_prime)
    elif tf == 'h':
        net.set_transfer_function(tanh)
        net.set_transfer_derivative(tanh_prime)
    
    return net
    
def main():
    print 'Loading MNIST data...'
    training_data, validation_data, test_data = ml.load_data_wrapper()
    training_data2 = [(x,np.argmax(y)) for x,y in training_data]
    n_train, n_val, n_test = (len(training_data), len(validation_data), len(test_data))
    
    net = prompt_for_net_params()
    eta = 0.5
    mbs = 10
    epoch = 0
    
    eta = float(raw_input('Enter learning rate (currently {}): '.format(eta)) or eta)
    mbs = int(raw_input('Enter mini-batch size (currently {}): '.format(mbs)) or mbs)

    show_menu()
    while True:
        ch = prompt_for_action()
        if ch == 'q':
            print 'Bye!'
            sys.exit(0)
        elif ch == 'h':
            show_menu()
        elif ch == 'r':
            net = prompt_for_net_params()
            show_menu()
            epoch = 0
        elif ch == 'l':
            eta = float(raw_input('Enter learning rate (currently {}): '.format(eta)) or eta)
        elif ch == 'm':
            mbs = int(raw_input('Enter mini-batch size (currently {}): '.format(mbs)) or mbs)
        elif ch == 'e':
            num_epochs = 10
            num_epochs = int(raw_input('Enter # of epochs [{}]: '.format(num_epochs)) or num_epochs)
            for e in xrange(num_epochs):
                net.SGD_Epoch(training_data, mbs, eta)
                epoch += 1
                fparams = (e+1, epoch, net.evaluate(validation_data), n_val)
                print 'Epoch {} ({}): {} / {} validation samples correctly classified'.format(*fparams)
        elif ch == 'n':
            print '{} / {} training samples correctly classified'.format(net.evaluate(training_data2), n_train)
        elif ch == 't':
            print '{} / {} testing samples correctly classified'.format(net.evaluate(test_data), n_test)

if __name__ == '__main__':
    main()
