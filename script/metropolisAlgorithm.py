import numpy as np
import graph 
import random as rnd

def metropolis(g):
    grouping = setupMetropolis(g)

    for i in range(g.n*(g.q)):
        rnd_point = rnd.randint(0,g.n-1)
        grouping = metropolisSwap(g,grouping,rnd_point)

    return grouping

def setupMetropolis(g:graph.SBM):
    group_dict = dict()
    for v in range(g.n):
        group_dict[v] = rnd.randint(0,g.q-1)
    return group_dict

def metropolisSwap(g:graph.SBM, group_dict:dict, v):
    ori_group = group_dict[v]
    new_group = ori_group
    while(new_group == ori_group):
        new_group = rnd.randint(0,g.q-1)

    deltaE = 2*sum([-1 if new_group== group_dict[u] else 1 for u in g.g.neighbors(v)])
    if(deltaE<0):
        group_dict[v] = new_group
        return group_dict

    # J/T = log(pin/pout)
    noise = np.log(g.p_in/(g.p_out))
    boltz = 1#1.38e-23
    probability = min(1,np.exp(-deltaE*noise/boltz))

    if(rnd.random()<probability):
        group_dict[v] = new_group

    return group_dict

