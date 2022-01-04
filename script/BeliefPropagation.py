import random as rnd
from networkx.algorithms.centrality import group
import numpy as np
from numpy.core.fromnumeric import product
import graph

def BP(g:graph.SBM):
    marginals_msg = np.zeros((g.n,g.n,g.q))
    for i in range(g.n):
        for j in range(g.n):
            for k in range(g.q):
                marginals_msg[i][j][k]= rnd.random()
            marginals_msg[i][j]=normalise(marginals_msg[i][j])


    for i in range(g.n):
        v = rnd.randint(0,g.n-1)
        marginals_msg = run(g,marginals_msg,v)

    marginals = finalMarginal(marginals_msg,g)
    grouping = extractGroups(marginals)

    return grouping

def run(g:graph.SBM, msg,v):
    for u in g.g.neighbors(v):
        for r in range(g.q):
            product = 1
            for j in g.g.neighbors(v):
                if(v!= j):
                    sum = 0
                    for k in range(g.q):
                        sum+=msg[j][u][k] * prior()
                    product *= sum
            msg[v][u][r] = product
        msg[v][u] = normalise(msg[v][u])

    return msg

def finalMarginal(msg,g):
    marginals = np.zeros((g.n,g.q))
    for i,v in enumerate(msg):
        
        for r in range(g.q):
            product = 1
            for u in v:
                sum=0
                for k in u:
                    sum+= k * prior()
                product *= sum
            marginals[i][r] = product
        
        marginals[i] = normalise(normalise[i])
    return marginals



def extractGroups(msg):
    groups = np.zeros(len(msg))
    for i in range(len(msg)):
        groups[i] = np.argmax(msg[i])
    
    return groups

def prior():
    return 1

def normalise(arr):
    norm = np.linalg.norm(arr)
    if(norm != 0):
        return arr/norm
    return arr