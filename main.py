import argparse
from neurosmash import Agent, Environment, Episode, QNetwork


def main(args):

    model = QNetwork(n_hidden = args.size * args.size, n_actions = 3)
    agent = Agent(model = model)
    env = Environment(args.ip, args.port, args.size, args.timescale)
    episode = Episode(env, agent)
    n_episodes_won = 0

    for i in range(args.n_episodes):
        episode.run()
        if episode.is_win:
            print(f"Agent won episode {i + 1}")
            n_episodes_won += 1
        else:
            print(f"Agent lost episode {i + 1}")
        agent.perturb_weights(model)

    print(f"Won/total: {n_episodes_won}/{args.n_episodes}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Set-up parameters
    p.add_argument('--ip', type=str, default="127.0.0.1", help="IP address that the TCP/IP interface listens to")
    p.add_argument('--port', type=int, default=13000, help="Port number that the TCP/IP interface listens to")
    p.add_argument('--size', type=int, default=96, help="Size of the environment's texture")
    p.add_argument('--timescale', type=float, default=10.0, help="Simulation speed (higher is faster)")

    # Simulation parameters
    p.add_argument('--n_episodes', type=int, default=10, help="Number of episodes we want to run")

    args = p.parse_args()

    main(args)
