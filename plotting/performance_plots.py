import matplotlib.pyplot as plt
import json
import numpy as np


def plot_rewards(filename):
    with open(filename, 'r') as openfile:
        output = json.load(openfile)


plot_rewards('./../logs/output.json')