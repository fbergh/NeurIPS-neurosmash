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
        wins = [agent.wins for agent in gen[gen_idx]]
        rewards = [agent.reward for agent in gen[gen_idx]]
        performance = {"generation": gen_idx,
                       "rewards": rewards,
                       "wins": wins}

        self.log["performance"].append(performance)

    def close(self):
        with open(os.path.join(self.path, self.filename), 'w') as outfile:
            json.dump(self.log, outfile, indent="  ")