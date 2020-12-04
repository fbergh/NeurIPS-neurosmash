from .agent import Agent
from .agent_utils import get_layer_weights, set_layer_weights
from networks import KaimingInit
import mxnet as mx
from mxnet import nd
import numpy as np

class EvolutionaryAgent(Agent):
    """ Evolutionary Strategy Agent class """
    def __init__(self, model, model_params, mutation_step, weights=None, ctx=mx.cpu()):
        super().__init__()
        self.mutation_step = mutation_step
        self.model = model(model_params)
        self.model.net.initialize(KaimingInit(), ctx=ctx, force_reinit=True)
        if weights:
            self.set_weights(weights)
        
    def step(self, end, reward, state):
        actions_probabilities = self.model(nd.array(state)).asnumpy()
        action = np.argmax(actions_probabilities)
        self.action_counter[action] += 1
        return action

    def get_weights(self):
        weights = []
        for layer in self.model.net:
            weights.append(get_layer_weights(layer))
        return weights

    def set_weights(self, weights):
        for i, layer in enumerate(self.model.net):
            set_layer_weights(layer, weights[i])