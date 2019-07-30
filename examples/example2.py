# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 09:35:30 2019

@author: constatza
"""

import preprocessor as pre
import finitesolver as fs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% read data
nodes = pre.read_nodes('cantilever_nodes.csv')
connectivity = pre.read_connectivity('cantilever_connectivity.csv')

#%% create model 
mystructure = fs.Model(nodes, connectivity)
mystructure.thickness = 0.01

#%% read stochastic field

stochastic_field = pd.read_csv('stochastic_field.csv', sep=',',header=None)
stochastic_field = stochastic_field.T
Emean = 30e6

Nsim = stochastic_field.shape[1]
Nelem = stochastic_field.shape[0]
height = np.linspace(0.5, 9.5, Nelem)
stochastic_field.set_index(height, inplace=True)

ux = []
for i in range(Nsim):  
    for element in mystructure.elements:
        element_height = element.mass_center[1]
        S = stochastic_field.loc[element_height, i] 
        E = Emean*(1+S)
        element.material = fs.ElasticIsotropic(young_modulus=E, poisson_ratio=.3)
    
    mystructure.assemble_stiffness_matrix()
    mystructure.solve()
    ux.append(mystructure.nodes.displacementsX.values[-1])    
    
ux = np.array(ux)    

plt.hist(ux, 30, density=True)
plt.show()
