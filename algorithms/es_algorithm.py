### IMPORTS ###

from .algorithm import Algorithm
import numpy as np
from mxnet import nd
from agents import EvolutionaryAgent
from processing import Logger


### EVOLUTIONARY STRATEGY ALGORITHM ###

class ESAlgorithm(Algorithm):
    def __init__(self, episode, model_type, model_params, min_mutation_step, initial_mutation_step, logger_filename):
        super().__init__(episode, logger_filename)
        self.model_type = model_type # Which type of model to use in the agents
        self.model_params = model_params # Parameters required to create models for new agents
        self.min_mutation_step = min_mutation_step # Minimum step size for mutations
        self.initial_mutation_step = initial_mutation_step # Initial step size for mutations

    def run(self, n_gens, gen_size, iter_per_agent, do_mutation=True, do_crossover=True):
        """ Run the evolutionary strategy algorithm for the given number of generations and generation size """
        # Initialize logger and table to store agents (+1 to account for gen 0)
        self.logger = Logger(self.logger_filename)
        self.generations = np.zeros((n_gens+1, gen_size)).astype(EvolutionaryAgent) 

        # Initialize the first generation as random evolutionary agents
        for i in range(gen_size):
            self.generations[0,i] = EvolutionaryAgent(self.model_type, self.model_params, self.initial_mutation_step)
        
        # For each generation, do the following:
        for gen in range(n_gens+1):
            print(f"\nRunning agents in generation {gen}")
            # Run each agent for the desired number of iterations
            for i, agent in enumerate(self.generations[gen]):
                print(f"Running agent {i+1}")
                self.run_agent(agent, iter_per_agent)
                print(f"Action proportions of agent {i+1}: {agent.action_proportions}")
                print(f"Agent {i+1} won {agent.total_wins} times (reward: {agent.total_reward:.3f})")
            # Print performance of current generation
            self.print_gen_performance(gen)
            # Log performance
            self.logger.log_gen_performance(self.generations, gen)
            # Save generation to a pickle file
            self.save_generation(self.generations, gen)
            # Generate the next generation if necessary
            if gen != n_gens:
                self.generations[gen+1] = self.create_generation(self.generations[gen], do_mutation, do_crossover)
        self.logger.close()

    def create_generation(self, prev_gen, do_mutation, do_crossover):
        """ Create a new generation of evolutionary agents """
        gen_size = len(prev_gen)
        new_gen = np.zeros(gen_size).astype(EvolutionaryAgent)
        for individual in range(gen_size):
            chosen_parents = self.get_parents(prev_gen, 2)
            new_gen[individual] = self.procreate(chosen_parents, do_mutation, do_crossover)
        return new_gen

    def get_parents(self, possible_parents, n_parents):
        """ Sample the required number of parents based on obtained rewards """
        parent_rewards = np.array([parent.total_reward for parent in possible_parents])
        probabilities = parent_rewards/sum(parent_rewards)
        return np.random.choice(possible_parents, size=n_parents, replace=False, p=probabilities)

    def procreate(self, parents, do_mutation, do_crossover):
        """ Generate new child from parents """
        # If crossover should occur, do it (otherwise, copy first parent)
        child_weights = self.crossover(parents) if do_crossover else parents[0].get_weights()
        # If mutation should occur, do it (otherwise, keep weights the same)
        mutation_step = self.get_mutation_step(parents)
        child_weights = self.mutation(child_weights, mutation_step) if do_mutation else child_weights
        # Return a new agent with the correct weights
        return EvolutionaryAgent(self.model_type, self.model_params, mutation_step, child_weights)

    def crossover(self, parents):
        """ Perform crossover by combining rows of network layers of all parents """
        # Initialize the new weights as the weights of the first parent
        new_weights = parents[0].get_weights().copy()

        # For each layer in the network, do the following:
        for layer in range(len(new_weights)):
            new_layer = new_weights[layer]
            # Randomly split the rows of the layer into approximately equally-sized groups, one for each parent
            indices = np.random.permutation(new_layer.shape[0])
            splits = np.array_split(indices, len(parents))

            # For each parent, replace the corresponding rows with the weights of the parent
            # Note: the first parent can be skipped, as their weights are already in new_layer
            for i in range(1,len(parents)):
                parent_layer = parents[i].get_weights()[layer]
                new_layer[splits[i]] = parent_layer[splits[i]]

            # Store the new layer weights
            new_weights[layer] = new_layer
        
        # Return the new set of weights obtained using crossover
        return new_weights

    def mutation(self, weights, mutation_step):
        """ Perform mutations on the given weights based on the mutation step size """
        new_weights = []
        for layer in weights:
            weight_mutations = mutation_step * nd.random_normal(0,1,shape=layer.shape)
            new_weights.append(layer + weight_mutations)
        return new_weights

    def get_mutation_step(self, parents):
        """ Compute mutation step based on the reward fraction of the parents """
        parent_rewards = [parent.total_reward for parent in parents]
        new_mutation_step = 1 / np.average(parent_rewards)
        return min(max(new_mutation_step, self.min_mutation_step), self.initial_mutation_step)
