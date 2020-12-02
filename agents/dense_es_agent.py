from .es_agent import ESAgent
from .agent_utils import get_layer_weights, set_layer_weights
from networks import KaimingInit, DenseNet
import mxnet as mx
from mxnet import nd
import numpy as np

class DenseESAgent(ESAgent):
    """
    Simple agent that takes the best action according to its network
    This agent has functions for getting and setting the weights in its netork
    """
    def __init__(self, params, mutation_step, weights=None, ctx=mx.cpu()):
        super().__init__(mutation_step)
        self.model = DenseNet(n_inputs=np.prod(params["img_shape"])*params["n_channels"], n_hidden=params["n_hidden"], n_actions=3)
        self.model.net.initialize(KaimingInit(), ctx=ctx, force_reinit=True)
        if weights:
            self.set_weights(weights)

    def step(self, end, reward, state):
        actions_probabilities = self.model(nd.array(state)).asnumpy()
        return np.argmax(actions_probabilities)

    def get_weights(self):
        weights = []
        for layer in self.model.net:
            weights.append(get_layer_weights(layer))
        return weights

    def set_weights(self, weights):
        for i, layer in enumerate(self.model.net):
            set_layer_weights(layer, weights[i])