### IMPORTS ###

from performance_plots import extract_reward_data, extract_win_data, extract_action_data, extract_mutation_data, plot_action_proportions
import matplotlib.pyplot as plt
import os
import numpy as np


### CONSTANTS ###

RANDOM = os.path.join("output", "logs", "random_log.json")
DENSE = os.path.join("output", "logs", "dense_log.json")
CONV = os.path.join("output", "logs", "conv_log.json")
LOG_LOCATIONS = [RANDOM, DENSE, CONV]
PLOT_LOCATION = os.path.join("output", "plots")


### PLOTTING COMPARISONS

def plot_average_rewards_comparison(reward_data):
    """ Plot average rewards for all agent types in a single plot """
    fig, ax = plt.subplots()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Reward")
    labels = ["Random", "Dense", "Convolutional"]
    for i, data in enumerate(reward_data):
        generation, average_reward, min_reward, max_reward = data
        ax.plot(generation, average_reward, label = labels[i])
        ax.fill_between(generation, min_reward, max_reward, alpha = 0.1)
        ax.set_xticks(generation, generation)
    ax.legend(loc="upper left")
    ax.grid()
    fig.savefig(os.path.join(PLOT_LOCATION, "average_rewards_comparison.png"))

def plot_cumulative_rewards_comparison(reward_data):
    """ Plot cumulative rewards for all agent types in a single plot """
    fig, ax = plt.subplots()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative Reward")
    labels = ["Random", "Dense", "Convolutional"]
    for i, data in enumerate(reward_data):
        generation, average_reward, _, _ = data
        cum_reward = np.cumsum(average_reward)
        ax.plot(generation, cum_reward, label = labels[i])
        ax.set_xticks(generation, generation)
    ax.legend(loc="upper left")
    ax.grid()
    fig.savefig(os.path.join(PLOT_LOCATION, "cumulative_rewards_comparison.png"))

def plot_action_proportions_comparison(action_data):
    """ Plot action proportions for all agent types (in separate plots) """
    labels = ["Random", "Dense", "Convolutional"]
    for i,data in enumerate(action_data):
        agent_type = labels[i]
        plot_action_proportions(data, agent_type)

def plot_mutation_steps_comparison(mutation_data):
    """ Plot mutations step sizes for all agent types in a single plot """
    fig, ax = plt.subplots()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Mutation Step Size")
    labels = ["Random", "Dense", "Convolutional"] 
    for i, data in enumerate(mutation_data):
        generation, average_mutation_step, min_mutation_step, max_mutation_step = data
        ax.plot(generation[1:], average_mutation_step[1:], label=labels[i])
        ax.fill_between(generation[1:], min_mutation_step[1:], max_mutation_step[1:], alpha = 0.1)
        ax.set_xticks(generation, generation)
    ax.legend(loc="upper left")
    ax.grid()
    fig.savefig(os.path.join(PLOT_LOCATION, "average_mutation_step_comparison.png"))


if __name__ == "__main__":
    if not os.path.exists(PLOT_LOCATION):
        os.mkdir(PLOT_LOCATION)

    reward_data = [extract_reward_data(filename) for filename in LOG_LOCATIONS]
    win_data = [extract_win_data(filename) for filename in LOG_LOCATIONS]
    action_data = [extract_action_data(filename) for filename in LOG_LOCATIONS]
    mutation_data = [extract_mutation_data(filename) for filename in LOG_LOCATIONS]

    plot_average_rewards_comparison(reward_data)
    plot_cumulative_rewards_comparison(reward_data)
    plot_action_proportions_comparison(action_data)
    plot_mutation_steps_comparison(mutation_data)