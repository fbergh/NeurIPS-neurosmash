from mxnet import nd, gluon
from mxnet.gluon import nn

class ConvNet(gluon.nn.Block):
    def __init__(self, n_channels, kernel_size, n_actions, **kwargs):
        super(ConvNet, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(
            nn.Conv2D(channels=n_channels, kernel_size=kernel_size, activation = 'relu'),
            nn.Conv2D(channels=n_channels * 2, kernel_size=kernel_size, activation = 'relu'),
            nn.Conv2D(channels=n_channels * 2, kernel_size=kernel_size, activation = 'relu'),
            nn.AvgPool2D(),
            nn.Dense(100),
            nn.Dense(units=n_actions)
        )

    def forward(self, state):
        # Convert state shape (w, h, c) to (c, w, h)
        state = nd.transpose(state, axes=(2, 0, 1))
        # Add extra batch dimension (1, c, w, h)
        state = state.expand_dims(0)
        # Return softmax of network output
        return nd.softmax(self.net(state))