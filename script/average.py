import numpy as np
from numpy.core.fromnumeric import argmax
import graph 
import random as rnd

def average(g):
    grouping = setupAverage(g)

    count = 0
    finished = 0
    iter = 0
    while(not finished):
        iter+=1
        rnd_point = rnd.randint(0,g.n-1)
        grouping,converged = averageSwap(g,grouping,rnd_point)

        if(converged):
            count +=1
            finished = count>g.n*2
        else:
            count= 0
    #print("iterations =", iter- g.n)
    return grouping

def setupAverage(g:graph.SBM):
    group_dict = dict()
    for v in range(g.n):
        group_dict[v] = rnd.randint(0,g.q-1)
    return group_dict

def averageSwap(g:graph.SBM, group_dict:dict, v):
    ori_group = group_dict[v]
    surrounding = [group_dict[u] for u in g.g.neighbors(v)]
    flag =True
    if(surrounding):
        modal_group = mode(surrounding)
        group_dict[v] = modal_group
        flag = ori_group== modal_group
        
    return group_dict, flag

def mode(lst):
    groups, counts = np.unique(lst,return_counts=True)
    index = argmax(counts)
    return groups[index]
