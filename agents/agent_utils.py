### IMPORTS ###

import mxnet as mx


### AGENT UTILITY FUNCTIONS ###

def set_layer_weights(layer, weights):
    # Set the weights of a given layer to the given weights
    if not type(layer) == mx.gluon.nn.conv_layers.AvgPool2D:
        initializer = mx.initializer.Constant(weights)
        layer.initialize(initializer, force_reinit=True)

def get_layer_weights(layer):
    # Retrieve the weights of a given layer
    if type(layer) == mx.gluon.nn.conv_layers.AvgPool2D:
        return mx.nd.zeros((100,100)) # This resolves the issue that AvgPool2D has no weights
    return layer.weight.data()