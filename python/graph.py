import random as rnd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time

from networkx.algorithms.smallworld import sigma
import numpy

class graph:
    
    def __init__(self,c_in,c_out,num_groups, size):
        self.num_groups = num_groups
        self.size = size
        self.p_in = c_in / size
        self.p_out = c_out / size
        print("P in = " + str(self.p_in) + ": p out = " + str(self.p_out))
        print("E[degree] = " + str((c_in + (num_groups -1)*c_out) /num_groups))

    def generate(self):
        """Generates a stochastic block model
        """
        self.node_dict = dict()

        for i in range(self.size):
            rand = rnd.randint(1,self.num_groups)
            self.node_dict[i] = rand
        matrix = np.zeros((self.size,self.size))
        edges = [[] for i in range(self.size)]
        for i in range(self.size):
            for j in range(i,self.size):
                if(i!= j and ((self.node_dict[i] == self.node_dict[j] and rnd.random()<self.p_in) or (rnd.random()<self.p_out))):
                    edges[i].append(j)
                    edges[j].append(i)
                    matrix[i][j] = 1
                    matrix[j][i] = 1
        
        self.adj_matrix = matrix
        self.adj_list = edges

    def display_List(self):
        for i in range(len(self.adj_list)):
            print(i, ":  ", self.adj_list[i])

    def display(self):
        """create a plt figure displaying the graph
        """
        g= nx.from_numpy_array(self.adj_matrix)
        plt.figure(figsize=(10,10))
        nx.draw(g,with_labels = True)
        plt.show()

    def displayGrounTruth(self):
        plt.figure(figsize=(10,10))

        g= nx.from_numpy_array(self.adj_matrix)
        pos = nx.random_layout(g)
        group_colouring = [[rnd.random(),rnd.random(),rnd.random()] for i in range(self.num_groups)]
        color_map = [group_colouring[self.node_dict[i]-1] for i in range(self.size)]

        nx.draw(g,node_color=color_map,with_labels = True)
        plt.show()



sizes = [10]*1
print(sizes)
graphs = []
timeStart = time.time()
for s in sizes:
    G = graph(0.8*s,0.2*s,2,s)
    G.generate()
    G.display_List()
    G.displayGrounTruth()
    graphs.append(G)

#print("took ",(time.time() - timeStart)/20," seconds per graph")