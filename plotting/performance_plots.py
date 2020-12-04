import matplotlib.pyplot as plt
import json
import numpy as np


def plot_rewards(filename):
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

    plt.plot(generation, average_reward)
    plt.fill_between(generation, min_reward, max_reward, alpha = 0.1)
    plt.xlabel("Generation")
    plt.ylabel("Average Reward")
    plt.xticks(generation, generation)
    plt.savefig("./../plots/average_rewards.png")


plot_rewards('./../logs/output.json')