from .network import Network
from mxnet import nd
from mxnet.gluon import nn

class DenseNet(Network):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.net.add(
            nn.Dense(in_units=params["n_inputs"], units=params["n_hidden"], activation='relu'),
            nn.Dense(in_units=params["n_hidden"], units=params["n_actions"])
        )

    def forward(self, state):
        # If input is an image, flatten the image to a state vector
        if len(state.shape) == 3:
            state = state.flatten()
        # Add extra batch dimension
        state = state.expand_dims(0)
        # Return softmax of network output
        return nd.softmax(self.net(state))