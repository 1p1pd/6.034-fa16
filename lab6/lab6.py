# MIT 6.034 Lab 6: Neural Nets
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), Jake Barnwell (jb16), and 6.034 staff

from nn_problems import *
from math import e
INF = float('inf')

#### NEURAL NETS ###############################################################

# Wiring a neural net

nn_half = [1]

nn_angle = [2, 1]

nn_cross = [2, 2, 1]

nn_stripe = [3, 1]

nn_hexagon = [6, 1]

nn_grid = [4, 2, 1]

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    return x >= threshold

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1 / (1 + e ** (-steepness * (x - midpoint)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return (x > 0) * x

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5 * (desired_output - actual_output) ** 2

# Forward propagation

def node_value(node, input_values, neuron_outputs):  # STAFF PROVIDED
    """Given a node, a dictionary mapping input names to their values, and a
    dictionary mapping neuron names to their outputs, returns the output value
    of the node."""
    if isinstance(node, basestring):
        return input_values[node] if node in input_values else neuron_outputs[node]
    return node  # constant input, such as -1

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""
    output = {}
    for neur in net.topological_sort():
        t = 0
        for node in net.get_incoming_neighbors(neur):
            t += node_value(node, input_values, output) * net.get_wires(node, neur)[0].get_weight()
        output[neur] = threshold_fn(t)
    return (output[net.get_output_neuron()], output)

# Backward propagation warm-up
def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    max = -INF
    result = None
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if max < func(inputs[0] + i * step_size, inputs[1] + j * step_size, inputs[2] + k * step_size):
                    max = func(inputs[0] + i * step_size, inputs[1] + j * step_size, inputs[2] + k * step_size)
                    result = [inputs[0] + i * step_size, inputs[1] + j * step_size, inputs[2] + k * step_size]
    return (max, result)

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    result = set([wire.endNode, wire.startNode, wire])
    for node in net.get_outgoing_neighbors(wire.endNode):
        result = result.union(get_back_prop_dependencies(net, net.get_wires(wire.endNode, node)[0]))
    return result

# Backward propagation
def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    result = {}
    network = net.topological_sort()
    network.reverse()
    for node in network:
        if net.is_output_neuron(node):
            result[node] = neuron_outputs[node] * (1 - neuron_outputs[node]) * (desired_output - neuron_outputs[net.get_output_neuron()])
        else:
            sum = 0
            for w in net.get_wires(node):
                sum += w.get_weight() * result[w.endNode]
            result[node] = neuron_outputs[node] * (1 - neuron_outputs[node]) * sum
    return result


def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    deltaB = calculate_deltas(net, desired_output, neuron_outputs)
    outputs = forward_prop(net, input_values, sigmoid)[1]
    for w in net.get_wires():
        w.set_weight(w.get_weight() + r * node_value(w.startNode, input_values, outputs) * deltaB[w.endNode])
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    count = 0
    (output, outputs) = forward_prop(net, input_values, sigmoid)
    while accuracy(desired_output, output) < minimum_accuracy:
        net = update_weights(net, input_values, desired_output, outputs, r)
        count += 1
        (output, outputs) = forward_prop(net, input_values, sigmoid)
    return (net, count)

# Training a neural net

ANSWER_1 = 41
ANSWER_2 = 20
ANSWER_3 = 8
ANSWER_4 = 124
ANSWER_5 = 57

ANSWER_6 = 1
ANSWER_7 = 'checkerboard'
ANSWER_8 = ['small', 'medium', 'large']
ANSWER_9 = 'B'

ANSWER_10 = 'D'
ANSWER_11 = ['A', 'C']
ANSWER_12 = ['A', 'E']


#### SURVEY ####################################################################

NAME = 'Yifan Wang'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 2
WHAT_I_FOUND_INTERESTING = "Everything"
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = None
