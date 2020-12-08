### IMPORTS ###

from .algorithm import Algorithm
from processing import Logger
from agents import RandomAgent
import numpy as np


### RANDOM ALGORITHM CLASS ###

class RandomAlgorithm(Algorithm):
    def __init__(self, episode, iter_per_agent, logger_filename):
        super().__init__(episode, iter_per_agent, logger_filename)

    def run(self, n_gens, gen_size):
        """ Run random agents for the given number of generations and generation size """

        # Initialize logger and table to store agents (+1 to account for gen 0)
        self.logger = Logger(self.logger_filename)
        self.generations = np.zeros((n_gens+1, gen_size)).astype(RandomAgent) 

        # Initialize the first generation as random random agents
        self.generations[0] = self.create_generation(gen_size)
        
        # For each generation, do the following:
        for gen in range(n_gens+1):
            print(f"\nRunning agents in generation {gen}")
            # Run each agent for the desired number of iterations
            for i, agent in enumerate(self.generations[gen]):
                print(f"Running agent {i+1}")
                self.run_agent(agent, self.iter_per_agent)
                print(f"Action proportions of agent {i+1}: {agent.get_action_proportions()}")
                print(f"Agent {i+1} won {agent.total_wins} times (reward: {agent.total_reward:.3f})")
            # Print performance of current generation
            self.print_gen_performance(gen)
            self.logger.log_gen_performance(self.generations, gen)
            # Generate the next generation if necessary
            if gen != n_gens:
                self.generations[gen+1] = self.create_generation(gen_size)
        self.logger.close()

    def create_generation(self, gen_size):
        """ Create a new generation of random agents """
        new_gen = [RandomAgent() for i in range(gen_size)]
        return new_gen