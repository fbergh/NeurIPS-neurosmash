### AGENT SUPERCLASS ###

class Agent(object):
    def __init__(self):
        self.rewards = []
        self.wins = []
        self.action_counter = {0: 0, 1: 0, 2: 0, 3:0}
        
    def step(self, end, reward, state):
        raise NotImplementedError

    def get_action_proportions(self):
        total_count = sum(self.action_counter.values())
        norm_action_counts = {action: count / total_count for action, count in self.action_counter.items()}
        return norm_action_counts

    @property
    def total_reward(self):
        return sum(self.rewards)
    
    @property
    def total_wins(self):
        return sum(self.wins)