import numpy as np
from neurosmash import Environment, Agent


# Based on https://openai.com/blog/evolution-strategies/
class EvolutionaryStrategy():

    def __init__(self, env):
        super().__init__()
        self.npop = 50 # population size
        self.sigma = 0.1 # noise standard deviation
        self.alpha = 0.001 # learning rate
        self.w = np.random.randn(3)  # initial policy
        self.env = env

    def step(self):
        N = np.random.randn(self.npop, 3) # workers with random seed
        R = np.zeros(self.npop) # returns
        for j in range(self.npop):
            w_try = self.w + self.sigma * N[j] # random weights for every population
            action = np.argmax(w_try) # pick action
            R[j] = self.env.step(action)[1] # reward for every population
        A = (R - np.mean(R)) / np.std(R) # normalise rewards
        self.w = self.w + self.alpha / (self.npop * self.sigma) * np.dot(N.T, A) # update weights


env = Environment()
es = EvolutionaryStrategy(env)
env.reset()

for i in range(100):
    es.step()
