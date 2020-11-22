import argparse
import time
import numpy as np
import mxnet.context as cuda
import matplotlib.pyplot as plt

from neurosmash import Environment, Episode
from network import DenseNet
from agent import SimpleESAgent, RandomAgent
from data import get_preprocess_transform
import algorithm


def main(args):
    img_size = args.size * args.size
    model = DenseNet(n_inputs=args.n_channels * img_size, n_hidden=args.n_hidden, n_actions=3)
    preprocess_transform = get_preprocess_transform(args.size, args.n_channels)
    env = Environment(args.ip, args.port, args.size, args.timescale, transform=preprocess_transform)
    agent_scores = np.zeros(args.n_agents)
    agents = np.zeros(args.n_agents, dtype=object)

    st_agent = time.time()
    for agent_id in range(args.n_agents):
        print(f"Time for one agent: {time.time() - st_agent} sec")
        st = time.time()
        agent = SimpleESAgent(model=model, ctx=args.device)
        print(f"Time init agent: {time.time() - st} sec")
        st = time.time()
        episode = Episode(env, agent, t_threshold=args.t_threshold, cooldown=args.cooldown)
        print(f"Time init episode: {time.time() - st} sec")
        n_episodes_won = 0
        total_rewards = 0
        st_episode = time.time()
        for i in range(args.n_episodes):
            print(f"Total time for resetting episode: {time.time() - st_episode} sec")
            st = time.time()
            is_win, end_reward = episode.run()
            print(f"Time for running episode: {time.time() - st} sec")
            if is_win:
                print(f"Agent {agent_id} won episode {i + 1}")
                n_episodes_won += 1
                total_rewards += end_reward
            else:
                print(f"Agent {agent_id} lost episode {i + 1}")
            st = time.time()
            agent.perturb_weights()
            print(f"Time to perturb weights: {time.time() - st} sec")
            st_episode = time.time()

        # Save agents and scores
        agent_scores[agent_id] = total_rewards
        agents[agent_id] = agent

        print(f"Won/total: {n_episodes_won}/{args.n_episodes}")
        print(f"Total agent score: {agent_scores[agent_id]}")
        st_agent = time.time()
    print(f"Best agent: {algorithm.pick_best_agent(agent_scores, agents)[0]}")


def str2bool(v):
    """
    For parsing booleans with argparse: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Network parameters
    p.add_argument('--n_channels', choices=["1", "2", "3"], default=3)
    p.add_argument('--n_hidden', type=int, default=128, help="Number of hidden units in network")

    # Set-up parameters
    p.add_argument('--ip', type=str, default="127.0.0.1", help="IP address that the TCP/IP interface listens to")
    p.add_argument('--port', type=int, default=13000, help="Port number that the TCP/IP interface listens to")
    p.add_argument('--size', type=int, default=96, help="Size of the environment's texture")
    p.add_argument('--timescale', type=float, default=10.0, help="Simulation speed (higher is faster)")

    # Simulation parameters
    p.add_argument('--n_episodes', type=int, default=10, help="Number of episodes we want to run")
    p.add_argument('--t_threshold', type=int, default=100, help="Number of timesteps one episode is allowed to run")
    p.add_argument("--cooldown", type=str2bool, nargs='?', const=True, default=False, help="Run episodes with cooldown")

    # Agent parameters
    p.add_argument('--n_agents', type=int, default=5, help="Number of agents")

    # Miscellaneous parameters
    p.add_argument('--device', type=str, default=cuda.gpu(0) if cuda.num_gpus() else cuda.cpu(),
                   help="Specifies on which device the neural network of the agent will be run")

    args = p.parse_args()
    print(args.device)

    main(args)
