from average import average
from glauberDynamics import glauber
from BeliefPropagation import BP
import matplotlib.pyplot as plt
from metropolisAlgorithm import metropolis
import graph
import numpy as np
import pandas as pd

columns=['n','q','p_in','p_out']
functions = [glauber,metropolis]
function_names =['Det/Glauber','Det/Metropolis']
iterations = 5
for i in range(iterations):
    columns.append(str(i+1))

def xsFormula(p_in,p_out):
    return p_in/(p_out)

def inverseXsFormula(x,p_out):
    return (x*p_out)

def generatePoints(samples = 51,n=1000):
    arr =  np.linspace(0,int(n),samples)[1:]
    return [ e/(n) for e in arr]

def generatePointsA(samples = 50,n=1000):
    arr =  np.linspace(0,int(n/2),samples)[1:]
    return [ e/(n) for e in arr]

def start(saveLoc = '_data.csv'):
    print('########START########')
    #setupFiles(saveLoc)

    points = generatePoints()
    points
    print(points)
    for p_out in points:
        for q in [5,2]:
            run(p_out,q,500,10 ,saveLoc=saveLoc)
            print("q=",q," complete")
        print(p_out, ' Completed')

    print('-------------DONE-----------------')

def covered(p_in,p_out,q,saveLoc):
    df = pd.read_csv(function_names[-1] + saveLoc)
    flag = ((df['p_in'] == p_in) & (df['p_out'] == p_out) & (df['q'] == q)).any()
    #print(q,' ', p_in, ' ', p_out)
    if flag: print(' covered')
    return flag

def setupFiles(saveLoc):
    data = []
    for i,f in enumerate(function_names):
        data.append(pd.DataFrame(columns=columns))
        data[i].to_csv(f+saveLoc,index=False)

def run(p_out,q,n, samples = 100,  saveLoc = '_data.csv'):

    #xs= np.zeros(samples)

    points = generatePoints()
    new_data = [[] for f in functions]
    for index,p_in in enumerate(points):
        print(index)
        if(p_in>p_out and (not covered(p_in,p_out,q,saveLoc))):
            #xs[index]=p_in
            
            save = {'n':n,'q':q,'p_in':p_in,'p_out':p_out}
            saveList = [save.copy() for f in functions]
            for k in range(iterations):
                #print(index,'.',k)
                g = graph.SBM(p_in,p_out,q,n)
                g.generate()
                g.set_q(2)

                for i,f in enumerate(functions):
                    grouping = f(g)
                    score = rating(grouping)
                    #score = g.rateModel(grouping)
                    saveList[i][str(k+1)] = score
  
            for i,s in enumerate(saveList):
                new_data[i].append(s)

    for i,d in enumerate(new_data):
        data= pd.read_csv(function_names[i]+saveLoc)
        new_df = pd.DataFrame(d,columns=columns)
        data = pd.concat([data,new_df])
        data.to_csv(function_names[i]+saveLoc,index=False)

    #data = pd.read_csv(saveLoc)
    #new_df = pd.DataFrame(new_data,columns=columns)
    #data =pd.concat([data,new_df])
    #data.to_csv(saveLoc,index=False)
    print('-----SAVED-----')
    #print(data)
    #print("analysis complete")

    #print(xs)

    #plt.plot(xs,glauber_ys,label='Glauber')
    #plt.plot(xs,metropolis_ys,label = 'Metropolis')
    #fig, axs = plt.subplots(2,1)
    #axs[0].set_title("accuracy")
    #axs[1].set_title("std")
    #for i in range(len(functions)):
    #    axs[0].plot(xs,ys[i])
    #    axs[1].plot(xs,ys_std[i],label=function_names[i])
    #plt.legend()
    #plt.show()

def rating(grouping:dict):
    total = sum(grouping.values()) / len(grouping)
    return max(total,1-total)

start()