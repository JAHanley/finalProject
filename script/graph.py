from math import perm
import random as rnd
import matplotlib.pyplot as plt
import pickle
import numpy as np
import networkx as nx
from itertools import permutations

class SBM:
    
    def __init__(self,p_in,p_out,num_groups, size):
        self.q = num_groups
        self.n = size
        self.p_in = p_in
        self.p_out = p_out
        self.generated = False

    def generate(self):
        """Generates a stochastic block model
        """
        self.generated = True
        self.node_dict = np.zeros(self.n)
        n=0
        while(True):
            n+=1
            for i in range(self.n):
                rand = rnd.randint(0,self.q-1)
                self.node_dict[i] = rand
            self.g = nx.Graph()
            self.g.add_nodes_from(range(self.n))
            for i in range(self.n):
                for j in range(i,self.n):
                    if(i!= j and ((self.node_dict[i] == self.node_dict[j] and rnd.random()<self.p_in) or (rnd.random()<self.p_out))):
                        self.g.add_edge(i,j)
            if(nx.is_connected(self.g)):
                return
            if(n>10000):
                print(n)
    
    def set_q(self,new_q):
        self.q = new_q

    def displayGroup(self,grouping):
        #ADD COLOURING N SHIT
        fig, (ax1,ax2) = plt.subplots(1,2,figsize = (20,10))
        group_colouring = [[rnd.random(),rnd.random(),rnd.random()] for i in range(self.q)]
        g= self.g
        pos = nx.fruchterman_reingold_layout(g)
        
        color_map = [group_colouring[grouping[i]-1] for i in range(self.n)]
        ground_truth_map = [group_colouring[self.node_dict[i]-1] for i in range(self.n)]
        fig.suptitle("Ground Truth vs Extracted communities")
        nx.draw(g,node_color=color_map,with_labels = True,ax= ax2,pos=pos)
        nx.draw(g,node_color=ground_truth_map,with_labels = True,ax= ax1, pos=pos)
        plt.show()
        
        print("score: ",self.rateModel(grouping))

    def rateModel(self,grouping):
        grouping_perm = permutations(range(self.q))
        top_score = -1
        best_perm = grouping
        for perm in grouping_perm:
            score = self.rateGrouping(grouping,perm)
            if(score > top_score):
                top_score = score
                best_perm = perm
        return top_score

    def rateGrouping(self,grouping,permutation):
        correct = sum([self.node_dict[i] ==permutation[grouping[i]] for i in range(self.n)])
        return np.round(correct/self.n,4)