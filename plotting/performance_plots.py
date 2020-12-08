### IMPORTS ###

import matplotlib.pyplot as plt
import json
import numpy as np


### CONSTANTS ###

LOG_LOCATION = "output/logs/output.json"
PLOT_LOCATION = "output/plots/"


### FUNCTIONS FOR PLOTTING ALGORITHM PERFORMANCE ###

def extract_reward_data(filename):
    with open(filename, 'r') as openfile:
        output = json.load(openfile)

    performance = output["performance"]
    generation = []
    min_reward = []
    max_reward = []
    average_reward = []

    for gen in performance:
        generation.append(gen["generation"])
        rewards = gen["rewards"]
        min_reward.append(np.min(rewards))
        max_reward.append(np.max(rewards))
        average_reward.append(np.average(rewards))

    return generation, average_reward, min_reward, max_reward

def extract_win_data(filename):
    with open(filename, 'r') as openfile:
        output = json.load(openfile)

    performance = output["performance"]
    generation = []
    average_wins = []

    for gen in performance:
        generation.append(gen["generation"])
        average_wins.append(np.average(gen["wins"]))

    return generation, average_wins

def plot_average_rewards(filename):
    generation, average_reward, min_reward, max_reward = extract_reward_data(filename)
    fig, ax = plt.subplots()
    ax.plot(generation, average_reward)
    ax.fill_between(generation, min_reward, max_reward, alpha = 0.1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Reward")
    ax.set_xticks(generation, generation)
    fig.savefig(PLOT_LOCATION+"average_rewards.png")

def plot_cumulative_rewards(filename):
    generation, average_reward, _, _ = extract_reward_data(filename)
    cum_reward = np.cumsum(average_reward)
    fig, ax = plt.subplots()
    ax.plot(generation, cum_reward)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative Reward")
    ax.set_xticks(generation, generation)
    fig.savefig(PLOT_LOCATION+"cumulative_rewards.png")

def plot_average_wins(filename):
    generation, average_wins = extract_win_data(filename)
    fig, ax = plt.subplots()
    ax.plot(generation, average_wins)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average #Wins")
    ax.set_xticks(generation, generation)
    fig.savefig(PLOT_LOCATION+"average_wins.png")

if __name__ == "__main__":
    plot_average_rewards(LOG_LOCATION)
    plot_cumulative_rewards(LOG_LOCATION)
    plot_average_wins(LOG_LOCATION)