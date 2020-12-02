class ESAgent(object):
    """
    Evolutionary Strategy Agent superclass
    """
    def __init__(self, mutation_step):
        self.mutation_step = mutation_step
        self.reward = 0
        self.wins = 0
        
    def step(self, end, reward, state):
        raise NotImplementedError