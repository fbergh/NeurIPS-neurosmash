### IMPORTS ###

from mxnet import nd, gluon
from mxnet.gluon import nn


### NEURAL NETWORK SUPERCLASS ###

class Network(nn.Block):
    def __init__(self, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.net = nn.Sequential()

    def forward(self, state):
        raise NotImplementedError