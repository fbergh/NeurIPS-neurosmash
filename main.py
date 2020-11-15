import argparse
import time

from neurosmash import Environment, Episode
from network import DenseNet
from agent import SimpleESAgent


def main(args):
    model = DenseNet(n_hidden=args.size * args.size, n_actions=3)
    agent = SimpleESAgent(model=model)
    env = Environment(args.ip, args.port, args.size, args.timescale)
    episode = Episode(env, agent, t_threshold=args.t_threshold, cooldown=args.cooldown)
    n_episodes_won = 0

    for i in range(args.n_episodes):
        episode.run()
        if episode.is_win:
            print(f"Agent won episode {i + 1}")
            n_episodes_won += 1
        else:
            print(f"Agent lost episode {i + 1}")
        start_time = time.time()
        agent.perturb_weights()
        print(f"Perturbing weights took {time.time() - start_time} sec")

    print(f"Won/total: {n_episodes_won}/{args.n_episodes}")


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

    args = p.parse_args()

    main(args)
