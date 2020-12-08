### IMPORTS ###

import numpy as np


### ALGORITHM SUPERCLASS ###

class Algorithm:
    def __init__(self, episode, iter_per_agent, logger_filename):
        self.episode = episode # Episode instance that can be used repeatedly for different agents
        self.iter_per_agent = iter_per_agent # Number of iterations per agent
        self.logger_filename = logger_filename # Filename for log file
        self.logger = None # Logger that stores results (initialized during run of algorithm)
        self.generations = None # List storing generations (initialized during run of algorithm)
        

    def run_agent(self, agent, n_iterations):
        """ Run a given agent for a given number of iterations, keeping track of rewards and wins """
        total_reward = 0
        n_wins = 0
        for i in range(n_iterations):
            print(f"Iteration {i+1}")
            is_win, end_reward = self.episode.run(agent)
            agent.wins.append(is_win)
            agent.rewards.append(end_reward)

    def print_gen_performance(self, gen_idx):
        # Print performance for a generation
        gen_wins = [agent.total_wins for agent in self.generations[gen_idx]]
        gen_rewards = [agent.total_reward for agent in self.generations[gen_idx]]
        print(f"Average wins in generation {gen_idx}: {np.average(gen_wins):.3f}")
        print(f"Average rewards in generation {gen_idx}: {np.average(gen_rewards):.3f}")
        print(f"Best agent in generation {gen_idx} won {gen_wins[np.argmax(gen_rewards)]} times (reward: {max(gen_rewards):.3f})")