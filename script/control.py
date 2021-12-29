from networkx.algorithms.centrality import group
from average import average
from glauberDynamics import glauber
import matplotlib.pyplot as plt
from metropolisAlgorithm import metropolis
import graph
import numpy as np
c_out = 0.1
samples=100
iterations = 5
q=2
n = q* 25
p_out = c_out / n
print(p_out)

def xsFormula(p_in,p_out):
    return p_in/(p_out)

def inverseXsFormula(x,p_out):
    return (x*p_out)

def generatePoints(samples = 10):
    return np.linspace(10,100,samples)

functions = [glauber, metropolis,average]
function_names =['glauber','metropolis','average']
ys = np.zeros((len(functions),samples))
ys_std = np.zeros((len(functions),samples))
#glauber_ys = np.zeros(samples)
#metropolis_ys = np.zeros(samples)
xs= np.zeros(samples)
scores= np.zeros(len(functions))
#glauber_scores = []
#metropolis_scores = []
points = generatePoints(samples)
print(len(points))

for index,x in enumerate(points):
    
    p_in = (x/100)
    c = (p_in*n + n*(q-1)*p_out)/q
    xs[index]=p_in
    #glauber_scores.append(0)
    #metropolis_scores.append(0)
    scores = np.zeros(len(functions))
    raw_scores = np.zeros((len(functions),iterations))
    for k in range(iterations):
        g = graph.SBM(p_in,p_out,q,n)
        g.generate()

        '''grouping = glauber(g)
        score = g.rateModel(grouping)
        glauber_scores[index] += score'''

        for i,f in enumerate(functions):
            grouping = f(g)
            score = g.rateModel(grouping)
            raw_scores[i,k] = score
            scores[i] += score
    
    for i in range(len(functions)):
        ys[i][index] = scores[i] / iterations
        ys_std[i][index] = np.max(raw_scores[i]) - np.min(raw_scores[i])

    print("row ", index," complete")

print("analysis complete")

print(xs)

#plt.plot(xs,glauber_ys,label='Glauber')
#plt.plot(xs,metropolis_ys,label = 'Metropolis')
fig, axs = plt.subplots(2,1)
axs[0].set_title("accuracy")
axs[1].set_title("std")
for i in range(len(functions)):
    axs[0].plot(xs,ys[i])
    axs[1].plot(xs,ys_std[i],label=function_names[i])
plt.legend()
plt.show()

