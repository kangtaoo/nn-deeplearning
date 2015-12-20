import mnist_loader
import network
import time
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
start_time = time.time();

net.SGD(training_data, 30, 10, 0.3, test_data=test_data)

print "------it takes %d seconds to complete the whole peocess-----" %(time.time() - start_time)