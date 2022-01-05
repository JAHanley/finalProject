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
    priors = np.ones(g.q)


    for i in range(g.n):
        priors = generate_priors(g,marginals_msg,priors)
        print(priors)
        v = rnd.randint(0,g.n-1)
        marginals_msg = run(g,marginals_msg,v,priors)

    priors = generate_priors(g,marginals_msg,priors)
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
                        sum+=msg[j][u][k] * priors[k]
                    product *= sum
            msg[u][v][r] = product
        msg[u][v] = normalise(msg[u][v])

    return msg

def finalMarginal(msg,g:graph.SBM,priors):
    marginals = np.zeros((g.n,g.q))
    for i,v in enumerate(msg):
        
        for r in range(g.q):
            product = 1
            for j in g.g.neighbors(i):
                u = v[j]
                print("-<>-",u)
                sum=0
                for i,k in enumerate(u):
                    sum+= k * priors[i]
                product *= sum
            marginals[i][r] = product
            print("----> ",marginals[i][r])
        
        marginals[i] = normalise(marginals[i])
    return marginals



def extractGroups(msg):
    groups = dict()
    for i in range(len(msg)):
        groups[i] = int(np.argmax(msg[i]))
    
    return groups

def generate_priors(g,msg,priors):
    print('Prior', priors)
    marginals = finalMarginal(msg,g,priors)
    print('--------------------------------')
    grouping = extractGroups(marginals)
    print(grouping)
    group_count = np.zeros(g.q)
    for i in range(g.n):
        group_count[int(grouping[i])] += 1
    
    return np.power(np.e,-group_count/(g.n/g.q))

def normalise(arr):
    if(not np.any(arr)):
        return np.ones(arr.shape)

    norm = np.linalg.norm(arr)
    if(norm != 0):
        arr = arr/norm
        if(not np.any(arr)):
            return np.ones(arr.shape)
        return arr
    return arr