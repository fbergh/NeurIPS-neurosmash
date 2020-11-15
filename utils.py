import mxnet as mx


def initialize_weights(layer, new_weights):
    initializer = mx.initializer.Constant(new_weights)
    layer.initialize(initializer, force_reinit=True)


def get_weights(layer):
    return layer.weight.data()
