# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import preprocessor as pre
import finitesolver as fs
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

#%% read data
nodes = pre.read_nodes('cantilever_nodes.csv')
connectivity = pre.read_connectivity('cantilever_connectivity.csv')

#%% create model 
mystructure = fs.Model(nodes, connectivity)
mystructure.thickness = 0.01

#%% material sambles

def logn(x, size):
    m = x
    v = (x/5)**2
    mu = np.log(m**2/(v + m**2)**.5)
    sigma = (np.log(v/m**2 + 1))**.5
    Es = np.random.lognormal(mean=mu, sigma=sigma, size=size)   
    return Es


Es = logn(30, 1000)

ux = []
def one_model(E):
    mystructure.material = fs.ElasticIsotropic(young_modulus=E, poisson_ratio=.3)
    mystructure.assemble_stiffness_matrix()
    mystructure.solve()
    ux.append(mystructure.nodes.displacementsX.values[-1])
    

ux_last = np.array(ux)    
for E in Es:
    one_model(E)

plt.plot(ux)
plt.show()
print(ux)