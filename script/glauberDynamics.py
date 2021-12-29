import numpy as np
import graph 
import random as rnd

def glauber(g):
    grouping = setupGlauber(g)

    for i in range(g.n):
        rnd_point = rnd.randint(0,g.n-1)
        grouping = glauberSwap(g,grouping,rnd_point)

    return grouping

def setupGlauber(g:graph.SBM):
    group_dict = dict()
    for v in range(g.n):
        group_dict[v] = rnd.randint(0,g.q-1)
    return group_dict

def glauberSwap(g:graph.SBM, group_dict:dict, v):
    ori_group = group_dict[v]
    new_group = ori_group
    while(new_group == ori_group):
        new_group = rnd.randint(0,g.q-1)

    deltaE = 2*sum([-1 if new_group== group_dict[u] else 1 for u in g.g.neighbors(v)])
    # J/T = log(pin/pout)
    noise = np.log(g.p_in/(g.p_out))

    probability = 0.5*(1-np.tanh(deltaE*noise*0.5))

    if(rnd.random()<probability):
        group_dict[v] = new_group

    return group_dict

