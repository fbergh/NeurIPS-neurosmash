import mxnet.ndarray as nd
import numpy as np
import socket
from PIL import Image

class Environment:
    def __init__(self, ip="127.0.0.1", port=13000, size=768, timescale=1, preprocessor=None):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.size = size
        self.timescale = timescale
        self.preprocessor = preprocessor

        self.client.connect((ip, port))

    def reset(self):
        self._send(1, 0)
        return self._receive()

    def step(self, action):
        self._send(2, action)
        end, reward, state = self._receive()
        return end, reward, state

    def state2image(self, state):
        return Image.fromarray(np.array(state, "uint8").reshape(self.size, self.size, 3))

    def _receive(self):
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end = data[0]
        reward = data[1]
        state = [data[i] for i in range(2, len(data))]
        if self.preprocessor:
            state = self.state2image(state)
            state = self.preprocessor.preprocess(nd.array(state))
        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))