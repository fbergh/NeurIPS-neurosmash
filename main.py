import argparse
import time
import numpy as np

from neurosmash import Environment, Episode
from network import DenseNet
from agent import SimpleESAgent, RandomAgent
import algorithm


def main(args):
    img_size = args.size * args.size
    model = DenseNet(n_inputs=3 * img_size, n_hidden=img_size, n_actions=3)
    env = Environment(args.ip, args.port, args.size, args.timescale)
    agent_scores = np.zeros(args.n_agents)
    agents = np.zeros(args.n_agents, dtype=object)

    for agent_id in range(args.n_agents):
        agent = SimpleESAgent(model=model)
        episode = Episode(env, agent, t_threshold=args.t_threshold, cooldown=args.cooldown)
        n_episodes_won = 0
        total_rewards = 0
        for i in range(args.n_episodes):
            episode.run()
            if episode.is_win:
                print(f"Agent {agent_id} won episode {i + 1}")
                n_episodes_won += 1
                total_rewards += episode.end_reward
            else:
                print(f"Agent {agent_id} lost episode {i + 1}")
            agent.perturb_weights() # Shouldn't we perturb weights before running all episodes?

        # Save agents and scores
        agent_scores[agent_id] = total_rewards 
        agents[agent_id] = agent

        print(f"Won/total: {n_episodes_won}/{args.n_episodes}")
        print(f"Total agent score: {agent_scores[agent_id]}")
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

    args = p.parse_args()

    main(args)
