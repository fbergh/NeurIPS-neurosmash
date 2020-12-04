import numpy as np
from mxnet import nd
from networks import DenseNet
from agents import ESAgent
from logger import Logger

class ESAlgorithm:

    def __init__(self, episode, agent_type, agent_params, iter_per_agent , mutation_lr, min_mutation_step, initial_mutation_step, filename):
        self.episode = episode # Episode instance can be used repeatedly for different agents
        self.agent_type = agent_type # Which type of agent to use
        self.agent_params = agent_params # Parameters required to create new agents
        self.mutation_lr = mutation_lr # Learning rate for mutations
        self.min_mutation_step = min_mutation_step # Minimum step size for mutations
        self.initial_mutation_step = initial_mutation_step # Initial step size for mutations
        self.filename = filename
        self.iter_per_agent = iter_per_agent 
        self.max_reward = iter_per_agent * 10

    def run(self, n_gens, gen_size, do_mutation=True, do_crossover=True):
        n_iters = self.iter_per_agent
        log = Logger(self.filename)
        do_mutation = False

        # Initialize table to store agents (+1 to account for gen 0)   
        self.generations = np.zeros((n_gens+1, gen_size)).astype(ESAgent) 

        # Initialize the first generation as random agents
        for i in range(gen_size):
            self.generations[0,i] = self.agent_type(self.agent_params, self.initial_mutation_step)
        
        # For each generation, do the following:
        for gen in range(n_gens+1):
            print(f"\nRunning agents in generation {gen}")
            # Run each agent for the desired number of iterations
            for i, agent in enumerate(self.generations[gen]):
                print(f"Running agent {i+1}")
                agent.reward, agent.wins = self.run_agent(agent, n_iters)
                print(f"Action proportions of agent {i+1}: {agent.get_action_proportions()}")
                print(f"Agent {i+1} won {agent.wins} times (reward: {agent.reward:.3f})")
            # Print performance of current generation
            self.print_performance(gen)
            log.log_performance(self.generations, gen)
            # Generate the next generation if necessary
            if gen != n_gens:
                self.generations[gen+1] = self.create_generation(self.generations[gen], do_mutation, do_crossover)
        log.close()


    def run_agent(self, agent, n_iterations):
        # Run a given agent for a given number of iterations, keeping track of reward and number of wins
        total_reward = 0
        n_wins = 0
        for i in range(n_iterations):
            print(f"Iteration {i+1}")
            is_win, end_reward = self.episode.run(agent)
            n_wins += is_win
            total_reward += end_reward
        return total_reward, n_wins

    def create_generation(self, prev_gen, do_mutation, do_crossover):
        # Create a new generation
        gen_size = len(prev_gen)
        new_gen = np.zeros(gen_size).astype(ESAgent)
        for individual in range(gen_size):
            chosen_parents = self.get_parents(prev_gen, 2)
            new_gen[individual] = self.procreate(chosen_parents, do_mutation, do_crossover)
        return new_gen

    def get_parents(self, possible_parents, n_parents):
        # Sample the required number of parents from the given list of possible  
        # parents without replacement, with probabilities based on obtained rewards
        parent_rewards = np.array([parent.reward for parent in possible_parents])
        probabilities = parent_rewards/sum(parent_rewards)
        return np.random.choice(possible_parents, size=n_parents, replace=False, p=probabilities)

    def procreate(self, parents, do_mutation, do_crossover):
        # If crossover should occur, do it
        if do_crossover:
            child_weights = self.crossover(parents)
        # Otherwise, set the weights of the child to those of one of the parents
        else:
            child_weights = parents[0].get_weights()
        # If mutation should occur, do it
        mutation_step = self.get_mutation_step(parents)
        if do_mutation:
            child_weights = self.mutation(child_weights, mutation_step)
        # Return a new agent with the correct weights
        return self.agent_type(self.agent_params, mutation_step, child_weights)

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
        parent_rewards = [parent.reward for parent in parents]
        reward_fraction = self.max_reward / np.average(parent_rewards) 
        parent_mutation_steps = [parent.mutation_step for parent in parents]
        new_mutation_step = np.average(parent_mutation_steps) * np.exp(self.mutation_lr * np.random.normal(0,1)) * reward_fraction

        if new_mutation_step < self.min_mutation_step:
            new_mutation_step = self.min_mutation_step
            
        return new_mutation_step

    def print_performance(self, gen_idx):
        gen_wins = [agent.wins for agent in self.generations[gen_idx]]
        gen_rewards = [agent.reward for agent in self.generations[gen_idx]]
        print(f"Average wins in generation {gen_idx}: {np.average(gen_wins):.3f}")
        print(f"Average rewards in generation {gen_idx}: {np.average(gen_rewards):.3f}")
        print(f"Best agent in generation {gen_idx} won {gen_wins[np.argmax(gen_rewards)]} times (reward: {max(gen_rewards):.3f})")