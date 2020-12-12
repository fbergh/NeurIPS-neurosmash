### IMPORTS ###

import json
import os


### LOGGER CLASS ###

class Logger:
    def __init__(self, filename, path="output/logs"):
        self.filename = filename
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        self.log = {"performance": []}

    def log_gen_performance(self, gen, gen_idx):
        """ Log generation performance to log dictionary """
        wins = [agent.total_wins for agent in gen[gen_idx]]
        rewards = [agent.total_reward for agent in gen[gen_idx]]
        actions = [agent.action_proportions for agent in gen[gen_idx]]
        mutation_steps = [agent.mutation_step for agent in gen[gen_idx]]
        performance = {"generation": gen_idx,
                       "rewards": rewards,
                       "wins": wins,
                       "actions": actions,
                       "mutation_steps": mutation_steps}
        self.log["performance"].append(performance)

    def close(self):
        """ Dump performance data to json file """
        with open(os.path.join(self.path, self.filename), 'w') as outfile:
            json.dump(self.log, outfile, indent="  ")