import numpy as np


def pick_best_agent(agent_scores, agents):
    best_agent = np.argmax(agent_scores)

    return best_agent, agents[best_agent]