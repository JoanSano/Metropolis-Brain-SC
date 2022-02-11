import argparse
from tqdm import tqdm
import torch
import wandb
import matplotlib.pylab as plt

from utils.data import random_graph_gen, graph_dumper
from utils.paths import check_path
from algorithm import Metropolis

parser = argparse.ArgumentParser()
# Main parameters 
parser.add_argument('-S','--steps', default=100, type=int, help='Number of Monte Carlo steps to sample')
parser.add_argument('-B','--batch', default=1, type=int, help='Batch size')
parser.add_argument('-T','--temperature', default=1.25, type=float, help='Temperature or randomness')
# Device and directories
parser.add_argument('-D','--device', default='cuda', type=str, help='Device')
parser.add_argument('-F','--folder', default='results', type=str, help='Rsults folder')
args = parser.parse_args()

if __name__ == '__main__':
    wandb.init(project="MC-Evolved-Brain-Graphs", entity="joansano")
    wandb.config.update(args)

    if args.device.lower() == 'cuda':
        device = args.device
    else:
        device = 'cpu'

    out_dir = check_path(args.folder)
    pre_dir = check_path(args.folder+'/pre/')
    post_dir = check_path(args.folder+'/post/')
    subjects = set()
    for i in range(args.batch):
        subjects.add(str(i+1))

    # Metropolis Monte Carlo evolution
    graphs = random_graph_gen(size=30, batch=args.batch, to_torch=True)
    graph_dumper(pre_dir, graphs, subjects, suffix='initial')
    MC = Metropolis(graphs, T=args.temperature)
    for i in tqdm(range(args.steps)):
        # We do a step of the algorithm
        E_tot, E_node, E_batch, E_node_batch = MC.step(logs=False)
        wandb.log({"Energy of the batch per node": E_node_batch})
    evolved_graphs = MC.get_graphs()
    graph_dumper(post_dir, evolved_graphs, subjects)

    plt.figure(figsize=(20,15))
    plt.imshow(evolved_graphs[0])
    cbar = plt.colorbar(ticks=[-1,0,1])
    cbar.set_label('Spin', rotation=270)
    plt.tight_layout()
    plt.show()




