import random as rnd
from networkx.algorithms.centrality import group
import numpy as np
from numpy.core.fromnumeric import product
import graph

def BP(g:graph.SBM):
    marginals_msg = np.ones((g.n,g.n,g.q))
    for i in range(g.n):
        for j in g.g.neighbors(i):
            for k in range(g.q):
                marginals_msg[i][j][k]= rnd.random()
            marginals_msg[i][j]=normalise(marginals_msg[i][j])
    priors = generate_priors(g)


    for i in range(g.n*g.q):
        v = rnd.randint(0,g.n-1)
        marginals_msg = run(g,marginals_msg,v,priors)

    marginals = finalMarginal(marginals_msg,g,priors)
    grouping = extractGroups(marginals)

    return grouping

def run(g:graph.SBM, msg,v,priors):
    for u in g.g.neighbors(v):
        for r in range(g.q):
            product = 1
            for j in g.g.neighbors(v):
                if(v!= j):
                    sum = 0
                    for k in range(g.q):
                        sum+=msg[j][u][k] * priors[r][k]
                    product *= sum
            msg[u][v][r] = product
        msg[u][v] = normalise(msg[u][v])

    return msg

def finalMarginal(msg,g:graph.SBM,priors):
    marginals = np.zeros((g.n,g.q))
    for v in range(g.n):
        for j in range(g.q):

            product = 1
            for u in g.g.neighbors(v):
                sum=0
                for k in range(g.q):
                    sum+= msg[u][v][k]*priors[k][j]
                product *= sum
            marginals[v][j] = product

        marginals[v] = normalise(marginals[v])
    return marginals



def extractGroups(msg):
    groups = dict()
    for i in range(len(msg)):
        groups[i] = int(np.argmax(msg[i]))
    
    return groups

def generate_priors(g):
    priors = np.zeros((g.q,g.q))
    for i in range(g.q):
        for j in range(g.q):
            priors[i][j] = g.p_in if j==i else g.p_out
    return priors

def normalise(arr):
    if(not np.any(arr)):
        return np.ones(arr.shape)

    norm = sum(arr)
    if(norm != 0):
        arr = arr/norm
        if(not np.any(arr)):
            return np.ones(arr.shape)
        return arr
    return arr