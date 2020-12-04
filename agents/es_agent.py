class ESAgent(object):
    """
    Evolutionary Strategy Agent superclass
    """
    def __init__(self, mutation_step):
        self.mutation_step = mutation_step
        self.reward = 0
        self.wins = 0
        self.action_counter = {0: 0, 1: 0, 2: 0}
        
    def step(self, end, reward, state):
        raise NotImplementedError

    def get_action_proportions(self):
        total_count = sum(self.action_counter.values())
        norm_action_counts = {action: count / total_count for action, count in self.action_counter.items()}
        return norm_action_counts
