from average import average
from glauberDynamics import glauber
from BeliefPropagation import BP
import matplotlib.pyplot as plt
from metropolisAlgorithm import metropolis
import graph
import numpy as np
import pandas as pd

columns=['n','q','p_in','p_out','Belief Propagation','Belief Propagation_STD','Glauber','Glauber_STD','Metropolis','Metropolis_STD','Average','Average_STD']

def xsFormula(p_in,p_out):
    return p_in/(p_out)

def inverseXsFormula(x,p_out):
    return (x*p_out)

def generatePoints(samples = 1000,n=10000):
    arr =  np.linspace(0,n,samples)[1:]
    return [ e/n for e in arr]

def start(saveLoc = 'data.csv'):
    data = pd.DataFrame(columns=columns)
    data.to_csv(saveLoc,index=False)
    points = generatePoints()

    for p_out in points:
        for q in [2,3,4,5]:
            run(p_out,q,q*20,1000,saveLoc=saveLoc)
            print("q=",q," complete")
        print(p_out, ' Completed')


def run(p_out,q,n, samples = 100, iterations = 10, saveLoc = 'data.csv'):
    c_out= n* p_out
    functions = [BP,glauber, metropolis,average]
    function_names =['Belief Propagation','Glauber','Metropolis','Average']
    ys = np.zeros((len(functions),samples))
    ys_std = np.zeros((len(functions),samples))
    #glauber_ys = np.zeros(samples)
    #metropolis_ys = np.zeros(samples)
    xs= np.zeros(samples)
    scores= np.zeros(len(functions))
    #glauber_scores = []
    #metropolis_scores = []
    points = generatePoints(samples+1)

    new_data = []
    for index,p_in in enumerate(points):
        if(p_in>p_out):
            c = (p_in*n + n*(q-1)*p_out)/q
            xs[index]=p_in
            
            scores = np.zeros(len(functions))
            raw_scores = np.zeros((len(functions),iterations))
            
            
            for k in range(iterations):
                g = graph.SBM(p_in,p_out,q,n)
                g.generate()

                for i,f in enumerate(functions):
                    grouping = f(g)
                    score = g.rateModel(grouping)
                    raw_scores[i,k] = score
                    scores[i] += score

            save = {'n':n,'q':q,'p_in':p_in,'p_out':p_out}
            for i in range(len(functions)):
                ys[i][index] = scores[i] / iterations
                save[str(function_names[i])] = ys[i][index]
                ys_std[i][index] = np.max(raw_scores[i]) - np.min(raw_scores[i])
                save[str(function_names[i]) + '_STD'] = ys_std[i][index]
            new_data.append(save)

    data = pd.read_csv(saveLoc)
    new_df = pd.DataFrame(new_data,columns=columns)
    data =pd.concat([data,new_df])
    data.to_csv(saveLoc,index=False)
    print('-----SAVED-----')
    print(data)
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

start()