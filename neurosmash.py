import mxnet.ndarray as nd
import numpy as np
import socket
from PIL import Image


class Environment:
    def __init__(self, ip="127.0.0.1", port=13000, size=768, timescale=1, width = 96, height = 96, preprocessor=None):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.port = port
        self.size = size
        self.timescale = timescale
        self.preprocessor = preprocessor
        self.width = width
        self.height = height

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
        if self.preprocessor is not None:
            state = self.state2image(state)
            state = self.preprocessor.preprocess(nd.array(state))
        return end, reward, state

    def _send(self, action, command):
        self.client.send(bytes([action, command]))


class Episode:
    def __init__(self, environment, agent, t_threshold=100, cooldown=False):
        self.env = environment
        self.agent = agent
        self.t_threshold = t_threshold
        self.cooldown = cooldown

    def run(self):
        end, reward, state = self.env.reset()

        # Run entire episode
        t = 0
        while not end:
            if t > self.t_threshold:
                self.env.reset()
                print("Time threshold reached. Stopping early.")
                break
            end, reward, state = self.step(reward, state)
            t += 1
        reward = self.determine_reward(reward,state, t)
        # We have won if the reward is 10, a loss might have reward < 10 but > 0
        is_win = reward == 10 
        end_reward = reward  # store reward at the end of episode

        # Additional steps if we want time for things to settle down
        if self.cooldown:
            for i in range(100):
                _ = self.step(reward, state)

        return is_win, end_reward

    def step(self, reward, state):
        action = self.agent.step(reward, state)
        end, reward, state = self.env.step(action)
        return end, reward, state

    # Determines whether Agent should get a small reward even though the Agent has lost
    def determine_reward(self, reward, state, time_elapsed):
        determined_reward = 5

        if (reward == 10): #agent has won..
            return reward
        else:
            return determined_reward * (time_elapsed/self.t_threshold)



