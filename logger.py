import json
import os


class Logger:
    def __init__(self, filename):
        self.filename = filename
        os.mkdir("logs")
        self.log = {"rewards": [], "wins": {}}

    def log_rewards(self, generation, agent, reward):
        rewards = {"generation": generation,
                   "agent": agent,
                   "reward": reward}

        self.log["rewards"].append(rewards)

    def close(self):
        with open(os.path.join("logs", self.filename), 'w') as outfile:
            json.dump(self.log, outfile, indent="  ")