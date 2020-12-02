from mxnet import nd, gluon
from mxnet.gluon import nn

class DenseNet(nn.Block):
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
        # Return softmax of network output
        return nd.softmax(self.net(state))