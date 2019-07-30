# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import preprocessor as pre
import finitesolver as fs
import matplotlib.pyplot as plt


np = fs.np
#%% read data
nodes = pre.read_nodes('cantilever_nodes.csv')
connectivity = pre.read_connectivity('cantilever_connectivity.csv')

#%% create model 
mystructure = fs.Model(nodes, connectivity)
mystructure.thickness = 0.01

#%% material sambles
size = 10000
m = 30e6
v = (m/5)**2
mu = np.log(m**2/(v + m**2)**.5)
sigma = (np.log(v/m**2 + 1))**.5
Es = np.random.lognormal(mean=mu, sigma=sigma, size=size)   


ux = []
for E in Es:
    mystructure.material = fs.ElasticIsotropic(young_modulus=E, poisson_ratio=.3)
    mystructure.assemble_stiffness_matrix()
    mystructure.solve()
    ux.append(mystructure.nodes.displacementsX.values[-1])
    

ux = np.asarray(ux)    


plt.hist(ux, 60, density=True)
plt.show()
