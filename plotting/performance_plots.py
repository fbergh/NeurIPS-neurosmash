import matplotlib.pyplot as plt
import json
import numpy as np
import os

def extract_reward_data(filename):
    # Extract reward data from json file
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
    # Extract win data from json file
    with open(filename, 'r') as openfile:
        output = json.load(openfile)

    performance = output["performance"]
    generation = []
    average_wins = []

    for gen in performance:
        generation.append(gen["generation"])
        average_wins.append(np.average(gen["wins"]))

    return generation, average_wins


def extract_action_data(filename):
    # Extract action proportion data from json file
    with open(filename, 'r') as openfile:
        output = json.load(openfile)

    performance = output["performance"]
    generation = []
    actions = []
    action_proportions = []

    for gen in performance:
        generation.append(gen["generation"])
        actions = list(gen["actions"][0].keys()) # Get all possible actions
        gen_actions = []

        # Get all action proportions for each agent in a generation
        for i in gen["actions"]:
            gen_actions.append(list(i.values()))

        # Get mean action proportions per generation across all agents
        action_proportions.append((np.mean(gen_actions, axis=0)))

    return generation, actions, action_proportions


def extract_mutation_data(filename):
    # Extract mutation data from json file
    with open(filename, 'r') as openfile:
        output = json.load(openfile)

    performance = output["performance"]
    generation = []
    min_mutation_step = []
    max_mutation_step = []
    average_mutation_step = []

    for gen in performance:
        generation.append(gen["generation"])
        mutation_steps = gen["mutation_steps"]
        min_mutation_step.append(np.min(mutation_steps))
        max_mutation_step.append(np.max(mutation_steps))
        average_mutation_step.append(np.average(mutation_steps))

    return generation, average_mutation_step, min_mutation_step, max_mutation_step


def plot_average_rewards(filename):
    # Plot average rewards across generations, including the minimal and maximal rewards per generation
    generation, average_reward, min_reward, max_reward = extract_reward_data(filename)
    fig, ax = plt.subplots()
    ax.plot(generation, average_reward)
    ax.fill_between(generation, min_reward, max_reward, alpha = 0.1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Reward")
    ax.set_xticks(generation, generation)
    fig.savefig(os.path.join("../", "plots/average_rewards.png"))

def plot_cumulative_rewards(filename):
    # Plot the cumulative average reward across generations
    generation, average_reward, _, _ = extract_reward_data(filename)
    cum_reward = np.cumsum(average_reward)
    fig, ax = plt.subplots()
    ax.plot(generation, cum_reward)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative Reward")
    ax.set_xticks(generation, generation)
    fig.savefig(os.path.join("../", "plots/cumulative_rewards.png"))


def plot_average_wins(filename):
    # Plot the average number of wins per generation
    generation, average_wins = extract_win_data(filename)
    fig, ax = plt.subplots()
    ax.plot(generation, average_wins)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average #Wins")
    ax.set_xticks(generation, generation)
    fig.savefig(os.path.join("../", "plots/average_wins.png"))


def plot_action_proportions(filename):
    # Plot the average action proportions per generation in a stacked bar plot
    generation, actions, action_proportions = extract_action_data(filename)
    action_proportions = np.asarray(action_proportions)

    fig, ax = plt.subplots()

    for i in range(action_proportions.shape[1]):
        if i == 0 :
            ax.bar(generation, action_proportions[:, i], label = f"Action {actions[i]}")
            previous_action_proportions = action_proportions[:, i]
        else:
            # stack on previous data
            ax.bar(generation, action_proportions[:, i], bottom=previous_action_proportions, label = f"Action {actions[i]}")
            previous_action_proportions += action_proportions[:, i]
            
    ax.legend()
    ax.set_xlabel("Generation")
    ax.set_ylabel("Action Proportions")
    ax.set_xticks(generation, generation)
    fig.savefig(os.path.join("../", "plots/action_proportions.png"))


def plot_mutation_steps(filename):
    # Plot the average, minimal, and maximal mutation step per generation
    generation, average_mutation_step, min_mutation_step, max_mutation_step = extract_mutation_data(filename)
    fig, ax = plt.subplots()
    generation, average_reward, _, _ = extract_reward_data(filename)
    ax.plot( average_mutation_step, label="Mutation")
    ax.fill_between(generation, min_mutation_step, max_mutation_step, alpha = 0.1)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Mutation Step Size")
    ax.set_xticks(generation, generation)
    fig.savefig(os.path.join("../", "plots/average_mutation_step.png"))


# Input file
performance_logs = os.path.join('../', "logs/output.json")


# Plotting
plot_average_rewards(performance_logs)
plot_cumulative_rewards(performance_logs)
plot_average_wins(performance_logs)
plot_action_proportions(performance_logs)
plot_mutation_steps(performance_logs)
