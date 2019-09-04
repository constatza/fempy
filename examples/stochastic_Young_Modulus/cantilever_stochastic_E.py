# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:30:38 2019

@author: constatza
"""

import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import fempy.fem.preprocessor as pre
import fempy.fem.postprocessor as post
numelX = 20
numelY = 50
Emean = 30
Nsim = 100

with open('analyzer.pkl', 'rb') as pickle_file:
    parent_analyzer = pickle.load(pickle_file)
model = parent_analyzer.provider.model


filepath = r"C:\Users\constatza\Documents\thesis\karhunenLoeve\stochastic_field.csv"
with open(filepath, 'rb') as stoch_file:
    stochastic_field = np.genfromtxt(stoch_file, delimiter=",")

if Nsim > stochastic_field.shape[0]:
    Nsim = stochastic_field.shape[0]    


E = Emean*(1+stochastic_field)    
U = np.empty((Nsim, 2100))
t1 = time.time()

for sim in range(Nsim):
    counter = -1
    listy = []
    for i in range(numelX): 
        for j in range(numelY):
            counter += 1
            # access elements bottom to top, ascending Y
            element = model.elements[counter] 
            element.material.young_modulus = E[sim, j]
            
    parent_analyzer.build_matrices()
    parent_analyzer.initialize()
    parent_analyzer.solve()
    U[sim, :] = parent_analyzer.linear_system.solution

print("Elapsed time = {:.3f} sec".format(time.time() - t1))
print("-------------------------")

model = post.assign_element_displacements(U[1,:], model)
ax = post.draw_deformed_shape(elements=model.elements, scale=200, color='g')
ax.set_aspect('equal', adjustable='box')
plt.draw()

np.save('stochastic_E_displacements_{:d}x{:d}'.format(numelX, numelY), U)