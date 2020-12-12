### IMPORTS ###

from neurosmash import Environment, Episode
from processing import Preprocessor
import argparse
import pickle
import os
import numpy as np


### CONSTANTS ###

GEN_LOCATION = os.path.join("output", "generations")
GEN_FILENAME = "generation50.pkl"
DEFAULT_CROP_RATIO = 28 / 96


### RUNNING DEMO ###

def run_demo(args):
    """ Run a demo of the best agent, as specified by the given arguments """
    # Initialize preprocessor, environment and episode
    crop_values = (0,0,0,int(DEFAULT_CROP_RATIO*args.size))
    preprocessor = Preprocessor(args.n_channels, crop_values)
    environment = Environment(args.ip, args.port, args.size, args.timescale, preprocessor)
    episode = Episode(environment, t_threshold=args.t_threshold, cooldown=args.cooldown)

    # Load best agent in specified generation file
    with open(os.path.join(args.gen_file_loc, args.gen_filename), "rb") as f:
        generation = pickle.load(f)
    gen_rewards = [agent.total_reward for agent in generation]
    best_agent = generation[np.argmax(gen_rewards)]

    # Run best agent for the specified number of iterations
    rewards, wins = [], []
    for i in range(args.n_iterations):
        print(f"Running iteration {i+1}")
        win, reward = episode.run(best_agent)
        rewards.append(reward)
        wins.append(win)
    print(f"The best agent in the given generation file won {sum(wins)} times (reward: {sum(rewards)})")
    print(f"Action proportions of this agent: {best_agent.action_proportions}")

def str2bool(v):
    """ Parse booleans obtained using argparse """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    return False

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Demo parameters
    p.add_argument("--n_iterations", type=int, default=5, help="How many times to run the best agent")
    p.add_argument("--gen_file_loc", type=str, default=GEN_LOCATION, help="Location of the generation pickle file")
    p.add_argument("--gen_filename", type=str, default=GEN_FILENAME, help="Filename of the generation pickle file")

    # Environment parameters
    p.add_argument('--ip', type=str, default="127.0.0.1", help="IP address that the TCP/IP interface listens to")
    p.add_argument('--port', type=int, default=13000, help="Port number that the TCP/IP interface listens to")
    p.add_argument('--size', type=int, default=40, help="Size of the environment's texture")
    p.add_argument('--timescale', type=float, default=10.0, help="Simulation speed (higher is faster)")
    p.add_argument('--n_channels', type=int, default=3, choices=[1,2,3], help="Number of channels to use")

    # Simulation parameters
    p.add_argument('--t_threshold', type=int, default=1000, help="Number of timesteps one episode is allowed to run")
    p.add_argument('--cooldown', type=str2bool, nargs='?', const=True, default=False, help="Run episodes with cooldown")

    args = p.parse_args()

    run_demo(args)