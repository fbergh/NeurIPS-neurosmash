### IMPORTS ###

from .agent import Agent
from networks import KaimingInit
import mxnet as mx
from mxnet import nd
import numpy as np


### EVOLUTIONARY STRATEGY AGENT CLASS ###

class EvolutionaryAgent(Agent):
    def __init__(self, model_type, model_params, mutation_step, weights=None, ctx=mx.cpu()):
        super().__init__()
        self.mutation_step = mutation_step
        self.model = model_type(model_params)
        self.model.net.initialize(KaimingInit(), ctx=ctx, force_reinit=True)
        if weights:
            self.set_weights(weights)
        
    def step(self, end, reward, state):
        actions_probabilities = self.model(nd.array(state)).asnumpy()
        action = np.argmax(actions_probabilities)
        self.action_counter[action] += 1
        return action

    def get_weights(self):
        # Retrieve all weights in the agent's model
        weights = []
        for layer in self.model.net:
            weights.append(self._get_layer_weights(layer))
        return weights

    def _get_layer_weights(self, layer):
        # Retrieve the weights of a given layer
        if type(layer) == mx.gluon.nn.conv_layers.AvgPool2D:
            return mx.nd.zeros((100,100)) # This resolves the issue that AvgPool2D has no weights
        return layer.weight.data()

    def set_weights(self, weights):
        # Set all weights in the agent's model
        for i, layer in enumerate(self.model.net):
            self._set_layer_weights(layer, weights[i])

    def _set_layer_weights(self, layer, weights):
        # Set the weights of a given layer to the given weights
        if not type(layer) == mx.gluon.nn.conv_layers.AvgPool2D:
            initializer = mx.initializer.Constant(weights)
            layer.initialize(initializer, force_reinit=True)