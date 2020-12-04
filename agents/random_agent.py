from .agent import Agent

class RandomAgent(Agent):
    """ Random Agent class """
    def __init__(self):
        super().__init__()

    def step(self, end, reward, state):
        self.action_counter[3] += 1
        return 3