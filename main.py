### IMPORTS ###

import argparse
import mxnet.context as cuda
from neurosmash import Environment, Episode
from algorithm import RandomAlgorithm, ESAlgorithm
from agents import EvolutionaryAgent
from networks import ConvNet, DenseNet
from processing import Preprocessor, Logger


### CONSTANTS ###

DEFAULT_CROP_RATIO = 28 / 96


### MAIN FUNCTION ###

def main(args):

    # If the chosen agent type is a random agent, run the agent for the desired number of generations and terminate
    if args.agent_type == "random":
        environment = Environment(args.ip, args.port, args.size, args.timescale)
        episode = Episode(environment, t_threshold=args.t_threshold, cooldown=args.cooldown)
        algorithm = RandomAlgorithm(episode, args.iter_per_agent, args.logger_filename)
        algorithm.run(args.n_generations, args.gen_size)
        return

    # Describe desired shape of states after preprocessing
    crop_values = (0,0,0,int(DEFAULT_CROP_RATIO*args.size))
    width = args.size - crop_values[0] - crop_values[2]
    height = args.size - crop_values[1] - crop_values[3]

    # Initialize preprocessor, environment and episode
    preprocessor = Preprocessor(args.n_channels, crop_values)
    environment = Environment(args.ip, args.port, args.size, args.timescale, preprocessor)
    episode = Episode(environment, t_threshold=args.t_threshold, cooldown=args.cooldown)
    
    # Initialize algorithm
    model_params = {"n_inputs":width*height*args.n_channels, 
                    "n_hidden":args.n_hidden, 
                    "n_channels":args.n_channels, 
                    "kernel_size":(args.kernel_size,args.kernel_size),
                    "n_actions":3}
    model_type = ConvNet if args.agent_type == "conv" else DenseNet
    algorithm = ESAlgorithm(episode, model_type, model_params, args.iter_per_agent, args.mutation_lr, 
                            args.min_mutation_step, args.initial_mutation_step, args.logger_filename)

    # Run algorithm
    algorithm.run(args.n_generations, args.gen_size,  args.do_mutation, args.do_crossover)

def str2bool(v):
    # Parse booleans obtained using argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    return False

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Set-up parameters
    p.add_argument('--ip', type=str, default="127.0.0.1", help="IP address that the TCP/IP interface listens to")
    p.add_argument('--port', type=int, default=13000, help="Port number that the TCP/IP interface listens to")
    p.add_argument('--size', type=int, default=40, help="Size of the environment's texture")
    p.add_argument('--timescale', type=float, default=10.0, help="Simulation speed (higher is faster)")

    # Simulation parameters
    p.add_argument('--t_threshold', type=int, default=1000, help="Number of timesteps one episode is allowed to run")
    p.add_argument('--cooldown', type=str2bool, nargs='?', const=True, default=False, help="Run episodes with cooldown")

    # Agent parameters
    p.add_argument('--agent_type', type=str, default="conv", choices=["random","conv","dense"], help="Type of agent to use (random/conv/dense)")
    p.add_argument('--n_hidden', type=int, default=1024, help="Number of hidden units (only applicable for agent type \"dense\")")
    p.add_argument('--n_channels', type=int, default=3, choices=[1,2,3], help="Number of channels to use (only applicable for agent type \"conv\")")
    p.add_argument('--kernel_size', type=int, default=3, help="Tuple denoting size of kernel (only applicable for agent type \"conv\")")

    # Algorithm parameters
    p.add_argument('--n_generations', type=int, default=50, help="Number of generations we want to run")
    p.add_argument('--gen_size', type=int, default=25, help="Number of agents per generation")
    p.add_argument('--iter_per_agent', type=int, default=10, help="Number of times each agent should be tested")
    p.add_argument('--mutation_lr', type=float, default=0.01, help="Learning rate for mutations")
    p.add_argument('--min_mutation_step', type=float, default=0.01, help="Minimum step size for mutations")
    p.add_argument('--initial_mutation_step', type=float, default=0.25, help="Initial step size for mutations")
    p.add_argument('--do_mutation', type=str2bool, nargs='?', const=True, default=True, help="Use mutation when creating children")
    p.add_argument('--do_crossover', type=str2bool, nargs='?', const=True, default=True, help="Use crossover when creating children")

    # Log parameters
    p.add_argument('--logger_filename', type=str, default="output.json", help="Filename of json output file")

    # Miscellaneous parameters
    p.add_argument('--device', type=str, default=cuda.gpu(0) if cuda.num_gpus() else cuda.cpu(),
                   help="Specifies on which device the neural network of the agent will be run")

    args = p.parse_args()

    main(args)
