import numpy as np
import socket
from PIL import Image
import mxnet as mx
from mxnet import autograd, gluon, nd, init
from mxnet.gluon import nn, Block
from mxnet.gluon.nn import LeakyReLU

class Agent:
    def __init__(self, model = None):
        self.model = model
        if self.model is not None:
            ctx = mx.cpu()
            self.model.initialize(ctx = ctx)
        pass

    def step(self, end, reward, state):
        # return 0 # nothing
        # return 1 # left
        # return 2 # right
        # return 3 # random
        if self.model is not None:
            actions = nd.softmax(self.model(nd.array(state))).asnumpy()
            return np.argmax(actions)
        else:
            return 3

    def perturb_weights(self, model, mean=0, sigma=0.05):
        for layer in model.net:
            # Collect current layer weights
            cur_weights = layer.weight.data()

            # Sample gaussian noise with same shape as layer
            layer_shape = cur_weights.shape
            gaussian_noise = nd.random_normal(mean, sigma, shape=layer_shape)

            # Add gaussian noise to current weights to retrieve new weights
            new_weights = cur_weights + gaussian_noise
            # print(new_weights - cur_weights)
            # Force re-initialization of layer weights
            self.initialize_weights(layer, new_weights)

    def initialize_weights(self, layer, new_weights):
        initializer = mx.initializer.Constant(new_weights)
        layer.initialize(initializer, force_reinit=True)


class QNetwork(gluon.nn.Block):
    def __init__(self, n_hidden, n_actions, **kwargs):
        super(QNetwork, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(
            nn.Dense(n_hidden, activation='relu'),
            nn.Dense(n_actions)
        )

    def forward(self, state):
        state = state.expand_dims(0)
        return self.net(state)


class Environment:
    def __init__(self, ip="127.0.0.1", port=13000, size=768, timescale=1):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.size = size
        self.timescale = timescale

        self.client.connect((ip, port))

    def reset(self):
        self._send(1, 0)
        return self._receive()

    def step(self, action):
        self._send(2, action)
        return self._receive()

    def state2image(self, state):
        return Image.fromarray(np.array(state, "uint8").reshape(self.size, self.size, 3))

    def _receive(self):
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end = data[0]
        reward = data[1]
        state = [data[i] for i in range(2, len(data))]

        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))


class Episode:
    def __init__(self, environment, agent, cooldown=False):
        self.env = environment
        self.agent = agent
        self.cooldown = cooldown
        self.is_win = False

    def run(self):
        end, reward, state = self.env.reset()

        # Run entire episode
        while not end:
            end, reward, state = self.step(end, reward, state)
        # We have won if the reward is greater than 0
        self.is_win = reward > 0

        # Additional steps if we want time for things to settle down
        if self.cooldown:
            for i in range(100):
                end, reward, state = self.step(end, reward, state)

    def step(self, end, reward, state):
        action = self.agent.step(end, reward, state)
        end, reward, state = self.env.step(action)
        return end, reward, state
