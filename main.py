import argparse

from utils.data import single_graph_gen
from algorithm import Metropolis

if __name__ == '__main__':
    graph = single_graph_gen(size=3)
    MC = Metropolis(graph)
    print(MC.get_graph())

    MC.__accept([1,2])
    print(MC.get_graph())

