import mnist_loader
import network
"""
Load data, set them into different sets:
training data, validation data and test data
"""
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

""" 
Initial the training network, specify the node number of input/hidden/output
"""
net = network.Network([784,30,10])

"""
Trigger the training/test process
Need to provice the epoch of training, mini-batch size and learning rate
"""
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)