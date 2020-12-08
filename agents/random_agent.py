### IMPORTS ###

from .agent import Agent


### RANDOM AGENT CLASS ###

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()
        self.mutation_step = 0 # For compatibility with logger

    def step(self, end, reward, state):
        self.action_counter[3] += 1
        return 3