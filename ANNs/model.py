import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

class Ising:
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs
        self.N = graphs.shape[-1]
        self.batch = graphs.shape[0]

    def graphs_energy(self):
        energy = torch.zeros((self.batch,))
        for i in range(self.N):
            for j in range(self.N):
                s = self.graphs[:,i,j]
                nb = self.graphs[:,(i+1)%self.N, j] + self.graphs[:,i,(j+1)%self.N] + self.graphs[:,(i-1)%self.N, j] + self.graphs[:,i,(j-1)%self.N]
                energy += -nb*s
        return energy/2.  # to compensate for over-counting

    def nodes_energy(self, s, pos):
        """ 
        Computes the energy associated with the nodes located in a specific location.
        Inputs:
            s: Value of the nodes' state
            pos: i,j coordinates of the node
        Outputs:
            -nb*s: Ising's energy of the nodes
        """
    
        i,j = pos
        nb = self.graphs[:,(i+1)%self.N, j] + self.graphs[:,i,(j+1)%self.N] + self.graphs[:,(i-1)%self.N, j] + self.graphs[:,i,(j-1)%self.N]
        return -nb*s

class ISINET(nn.Module):
    def __init__(self):
        super().__init__()
        # Given that an Ising model only deals with short-range interactions,
        #    small-sized 2D convolutions should do the trick.

class FREENET(nn.Module):
    def __init__(self) -> None:
        super().__init__()

def constrain():
    """
    Sets a constraint.
    """

    #TODO Add constrain
    pass

if __name__ == '__main__':
    pass