import numpy as np
import torch
from .paths import *
from subprocess import Popen, STDOUT, PIPE
import os
import re
import pandas as pd

def random_graph_gen(size=50, batch=1, states=[-1, 1], to_torch=False):
    graph = np.random.choice(states, size=(batch,size,size))
    if to_torch:
        graph = torch.tensor(graph)
    return graph    

def graph_dumper(data_path, graphs, subject_list, suffix='evolved'):
    for i, sub in enumerate(subject_list):
        name = sub + '_'+suffix+'.csv'
        gr = graphs[i]
        dataframe = pd.DataFrame(data=gr.numpy().astype(float))
        dataframe.to_csv(data_path+name, sep=',', header=False, float_format='%.6f', index=False)



if __name__ == '__main__':
    pass