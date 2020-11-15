from mxnet import autograd, gluon, nd, init
from mxnet.gluon import nn, Block


class DenseNet(gluon.nn.Block):
    def __init__(self, n_hidden, n_actions, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(
            nn.Dense(n_hidden, activation='relu'),
            nn.Dense(n_actions)
        )

    def forward(self, state):
        state = state.expand_dims(0)
        return nd.softmax(self.net(state))
