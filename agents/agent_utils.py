import mxnet as mx

def set_layer_weights(layer, weights):
    if not type(layer) == mx.gluon.nn.conv_layers.AvgPool2D:
        initializer = mx.initializer.Constant(weights)
        layer.initialize(initializer, force_reinit=True)

def get_layer_weights(layer):
    if type(layer) == mx.gluon.nn.conv_layers.AvgPool2D:
        return mx.nd.zeros((100,100))
    return layer.weight.data()