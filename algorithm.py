import numpy as np

class Ising:
    def __init__(self, graph):
        self.graph = graph
        self.N = graph.shape[0]

    def ising_energy(self):
        energy = 0
        for i in range(self.graph.shape[0]):
            for j in range(self.graph.shape[1]):
                s = self.graph[i,j]
                nb = self.graph[(i+1)%self.N, j] + self.graph[i,(j+1)%self.N] + self.graph[(i-1)%self.N, j] + self.graph[i,(j-1)%self.N]
                energy += -nb*s
        return energy/2.  # to compensate for over-counting

    def node_energy(self, s, pos):
        i,j = pos
        nb = self.graph[(i+1)%self.N, j] + self.graph[i,(j+1)%self.N] + self.graph[(i-1)%self.N, j] + self.graph[i,(j-1)%self.N]
        return -nb*s

class Metropolis():
    def __init__(self, graph, energy='Ising', T=1, states=[-1, 1]):
        # TODO 1: Evolution for 1 single graph
        # TODO 2: Make it work for a batch of brain graphs, not only one
        # TODO 3: Add FENET
        # TODO 4: Train FENET to learn the Ising energy function

        self.graph = graph
        self.N = graph.shape
        self.states = list(states)
        self.beta = 1./T
        if energy.lower == 'ising':
            self.Energy = Ising(graph)
        else:
            raise ValueError('Provide a valid energy function')

    def __flip(self, pos):
        i,j = pos
        s = int(self.graph[i,j])
        ns = -s#*np.random.choice(self.states) # The flip proposal might also be random -- useful if it has more than one direction
        dE = self.Energy.node_energy(ns, pos)-self.Energy.node_energy(s, pos)
        return ns, dE
    
    def __accept(self, pos):
        new, dE = self.__flip(pos)
        r = np.random.random()
        ratio = np.exp(-self.beta*dE)
        if dE < 0 or (r<ratio): 
            i,j = pos
            new_graph = self.graph
            self.graph[i,j] = new
            self.__init__(new_graph)
            
    def get_graph(self):
        return self.graph

if __name__ == '__main__':
    pass