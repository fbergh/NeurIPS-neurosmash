import matplotlib.pyplot as plt
import json
import numpy as np


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


def plot_average_rewards(filename):
    generation, average_reward, min_reward, max_reward = extract_reward_data(filename)
    plt.plot(generation, average_reward)
    plt.fill_between(generation, min_reward, max_reward, alpha = 0.1)
    plt.xlabel("Generation")
    plt.ylabel("Average Reward")
    plt.xticks(generation, generation)
    plt.savefig("./../plots/average_rewards.png")


def plot_cumulative_rewards(filename):
    generation, average_reward, _, _ = extract_reward_data(filename)
    cum_reward = np.cumsum(average_reward)
    plt.plot(generation, cum_reward)
    plt.xlabel("Generation")
    plt.ylabel("Cumulative Reward")
    plt.xticks(generation, generation)
    plt.savefig("./../plots/cumulative_rewards.png")


plot_average_rewards('./../logs/output.json')
plot_cumulative_rewards('./../logs/output.json')
