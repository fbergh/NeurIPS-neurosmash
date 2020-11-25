from mxnet import autograd, gluon, nd, init, random
from mxnet.gluon import nn, Block
import math


class KaimingInit(init.Initializer):
    """
    Class for Kaiming/He initalisation with MxNet, best used with ReLU (see link)
    (see: https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize/319849#319849)
    """
    def __init__(self, **kwargs):
        super(KaimingInit, self).__init__(**kwargs)

    def _init_weight(self, name, data):
        data[:] = random.normal(shape=data.shape)
        n_units_in = data.shape[0]
        data *= math.sqrt(2./n_units_in)


class DenseNet(gluon.nn.Block):
    def __init__(self, n_inputs, n_hidden, n_actions, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(
            nn.Dense(in_units=n_inputs, units=n_hidden, activation='relu'),
            nn.Dense(in_units=n_hidden, units=n_actions)
        )

    def forward(self, state):
        # If input is an image, flatten the image to a state vector
        if len(state.shape) == 3:
            state = state.flatten()
        # Add extra batch dimension
        state = state.expand_dims(0)
        return nd.softmax(self.net(state))

class ConvNet(gluon.nn.Block):
    def __init__(self, n_channels, kernel_size, n_actions, **kwargs):
        super(ConvNet, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(
            nn.Conv2D(channels = n_channels,kernel_size = kernel_size, activation='relu'),
            nn.Dense(units=n_actions)
        )

    def forward(self, state):
        state = state.expand_dims(0)
        return nd.softmax(self.net(state))