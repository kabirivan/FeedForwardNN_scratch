#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 20:38:56 2020

@author: aguasharo
"""

from random import seed
from random import random


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
def transfer_tanh(x):
	return (exp(x)-exp(-x))/(exp(x)+exp(-x))


def transfer_softmax(x):
    return np.exp(x)/np.sum(np.exp(x))