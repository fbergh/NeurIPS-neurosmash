from mxnet import init, random
import math

class KaimingInit(init.Initializer):
    """
    Class for Kaiming/He initalisation with MxNet, best used with ReLU (see link)
    (see: https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize/319849#319849)
    """
    def __init__(self, **kwargs):
        super(KaimingInit, self).__init__(**kwargs)

    def _init_weight(self, name, data):
        data[:] = random.normal(shape=data.shape)
        n_units_in = data.shape[0]
        data *= math.sqrt(2./n_units_in)