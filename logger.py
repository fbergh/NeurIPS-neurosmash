import json
import os


class Logger:
    def __init__(self, filename, path = "logs"):
        self.filename = filename
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        self.log = {"performance": []}

    def log_performance(self, gen, gen_idx):
        # Logs performance of agents across generations
        wins = [agent.wins for agent in gen[gen_idx]]
        rewards = [agent.reward for agent in gen[gen_idx]]
        actions = [agent.get_action_proportions() for agent in gen[gen_idx]]
        mutation_steps = [agent.mutation_step for agent in gen[gen_idx]]
        performance = {"generation": gen_idx,
                       "rewards": rewards,
                       "wins": wins,
                       "actions": actions,
                       "mutation_steps": mutation_steps}

        self.log["performance"].append(performance)

    def close(self):
        # Dump performance data to json file
        with open(os.path.join(self.path, self.filename), 'w') as outfile:
            json.dump(self.log, outfile, indent="  ")