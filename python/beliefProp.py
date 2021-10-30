from random import randint
import numpy as np
from python.graph import graph


size = input("size of network = ")
c_in = input("C_in = ")
c_out = input("C_out = ")
q = input("number of goups = ")



g = graph(c_in,c_out,q,size)
g.generate()

marginals = np.empty((size,q))

def beliefProp(adj_list, marginals, group_change_prob):
    '''runs a breadth first search with parameters adj_list, the graph, the marginals of each node
    and group_change_prob[i][j] which is the probabilities of changing from group i to group j
    '''
    starting_vertex = randint(0,len(g.adj_list))

    queue = [starting_vertex]
    explored_vertexes = set(starting_vertex)
    while not queue:
        v = queue.pop()
        for u in adj_list[v]:
            if(not (u in explored_vertexes)):
                explored_vertexes.add(u)
                queue.append(u)