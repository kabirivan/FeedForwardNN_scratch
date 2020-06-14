#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 20:38:56 2020

@author: aguasharo
"""

from random import seed
from random import random
import numpy as np


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network


# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation



# Transfer neuron activation

def transfer(x, type_function):
    
    if type_function == 'tanh':
        
        output = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        
    elif type_function == 'softMax': 
    
        output = np.exp(x)/np.sum(np.exp(x))
        
    else:
         
        output = x
        
    return output    

transfer_Funtions = ['None', 'tanh', 'softMax'] 

# Forward propagate input to a network output
def forward_propagate(network, row):
    transfer_Funtions = ['None', 'tanh', 'softMax'] 
    inputs = row
    cont = 0
    for layer in network:
        new_inputs = []
        
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation,transfer_Funtions[cont])
            new_inputs.append(neuron['output'])
           
        inputs = new_inputs
        cont = cont + 1
    return inputs


