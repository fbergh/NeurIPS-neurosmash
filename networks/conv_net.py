from .network import Network
from mxnet import nd
from mxnet.gluon import nn

class ConvNet(Network):
    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.net.add(
            nn.Conv2D(channels=params["n_channels"], kernel_size=params["kernel_size"], activation = 'relu'),
            nn.Conv2D(channels=params["n_channels"] * 2, kernel_size=params["kernel_size"], activation = 'relu'),
            nn.Conv2D(channels=params["n_channels"] * 2, kernel_size=params["kernel_size"], activation = 'relu'),
            nn.AvgPool2D(),
            nn.Dense(100),
            nn.Dense(units=params["n_actions"])
        )

    def forward(self, state):
        # Make sure that the input is an image
        assert len(state.shape) == 3, "Input to a ConvNet should be an image"
        # Convert state shape (w, h, c) to (c, w, h)
        state = nd.transpose(state, axes=(2, 0, 1))
        # Add extra batch dimension (1, c, w, h)
        state = state.expand_dims(0)
        # Return softmax of network output
        return nd.softmax(self.net(state))