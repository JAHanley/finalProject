import graph
from BeliefPropagation import BP

g= graph.SBM(0.8,0.1,2,20)
g.generate()

grouping = BP(g)
s=g.rateModel(grouping)
print("score = ",s)