import numpy as np
import torch

from ANNs.model import Ising, ISINET

class Metropolis():
    def __init__(self, graphs, energy='Ising', T=1, states=[-1, 1]):
        super().__init__()
        """ 
        It only works with 2D square graphs, which are essencially adjacency matrices.
        """
        # TODO 3: Add ISINET
        # TODO 4: Train ISINET to learn the Ising energy function
        # TODO 5: Work with adjacency matrices in the ising case (optional)
        # TODO 6: Add topological constraints and use the trained ISINET--> A[i,j] = 0 (e.g using a function)
        # TODO 7: Design FRENET and repeat the process

        self.graphs = graphs
        self.N = graphs.shape[-1]
        self.batch = graphs.shape[0]
        self.states = list(states)
        self.T = T
        self.beta = 1/self.T
        self.energy = energy
        if energy.lower() == 'ising':
            self.Energy = Ising(graphs)
        else:
            raise ValueError('Provide a valid energy function')

    def __flip(self, pos, random=True):
        """ 
        Propose a flip/change in the graphs at a specific location and
        and obtain the consequences of such flip.
        Inputs:
            pos: position at which to propose the flip
            random: The flip is random
        Outputs:
            ns: New states
            dE: Change in the energy of the graphs
        """

        i,j = pos
        s = self.graphs[:,i,j]
        if random:
            ns = s*np.random.choice(self.states) # The flip proposal might also be random -- useful if it has more than one direction
        else:
            ns = -s
        dE = self.Energy.nodes_energy(ns, pos)-self.Energy.nodes_energy(s, pos)
        return ns, dE
    
    def __accept(self, pos):
        """ 
        Accept or decline the proposed changes in the specified position.
        Inputs:
            pos: position at which to propose the flip
        Outputs:
            No outputs. The object is initialized with the new graphs.
        """

        new, dE = self.__flip(pos)
        new_graphs = self.graphs
        for gr in range(self.batch):
            r = np.random.random()
            ratio = np.exp(-self.beta*dE[gr])
            if dE[gr] < 0: 
                i,j = pos
                new_graphs[gr,i,j] = new[gr]
            elif r<ratio:
                i,j = pos
                new_graphs[gr,i,j] = new[gr]
            else:
                pass
        # We reinitialize the class with the updated graph
        self.__init__(new_graphs, T=self.T)

    def step(self, logs=False):
        """ 
        Performs a full Montecarlo update on the batch. Basically samples and accepts/declines a change
        in every location of the graph. 
        Inputs:
            None. It just uses the default graph, which is changing as "__acepts" dictates.
        Outputs:
            None. It updates the default graphs.
        """
        
        # Montecarlo Step
        for i in range(self.N):
            for j in range(self.N):
                self.__accept([i,j])  
        # Meaningufl metric/magnitudes to train
        if self.energy.lower() == 'ising':
            E_graphs = self.Energy.graphs_energy()
            E_node = E_graphs/(self.N**2)
            E_batch = torch.mean(E_graphs)
            E_node_batch = torch.mean(E_node)
            # Logs and returns
            if logs:
                print("Energy of the graphs: {}".format(E_graphs))
                print("Energy per node: {}".format(E_node))
                print("Energy of the batch: {}".format(E_batch))
            return E_graphs, E_node, E_batch, E_node_batch
        else:
            return 0, 0, 0, 0
            
    def get_graphs(self, b_gr=None):
        """ 
        Acces a specified graph inside the batch. 
        """

        if not b_gr:
            return self.graphs
        else:
            return self.graphs[b_gr]

if __name__ == '__main__':
    pass