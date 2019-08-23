# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""
import matplotlib.pyplot as plt
import numpy as np

import fem.preprocessor as pre
from fem.core.elements import Quad4
from fem.core.entities import DOFtype, Load
from fem.core.assemblers import ProblemStructural, ElementMaterialOnlyStiffnessProvider
from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.solvers import LinearSystem, SparseSolver
import fem.analyzers as analyzers

plt.close()
"""
Create rectangular mesh for the model.
"""
numelX = 20
numelY = 50
boundX = [0, 20]
boundY = [0, 50]
last_node = (numelX + 1) * (numelY + 1)

material = ElasticMaterial2D(stress_state=StressState2D.plain_stress,
                             young_modulus=3.76,
                             poisson_ratio=0.3779)

quad = Quad4(material=material, thickness=1)

model = pre.rectangular_mesh_model(boundX, boundY, numelX, numelY, quad)
#pre.draw_mesh(model.elements, color='b')

for node in model.nodes[:numelX+1]:
    node.constraints = [DOFtype.X, DOFtype.Y]

load1 = Load(magnitude=100, 
             node=model.nodes_dictionary[last_node-1], 
             DOF=DOFtype.X)


"""
Variable Young's modulus realization
"""
Nsim = 100
m = 30e6
v = (m/5)**2
mu = np.log(m**2/(v + m**2)**.5)
sigma = (np.log(v/m**2 + 1))**.5
Es = np.random.lognormal(mean=mu, sigma=sigma, size=Nsim)   
#plt.hist(Es,30)
#plt.show()

model.loads.append(load1)             
model.connect_data_structures()
linear_system = LinearSystem(model.forces)
solver = SparseSolver(linear_system)
provider = ProblemStructural(model)
provider.stiffness_provider = ElementMaterialOnlyStiffnessProvider() 

U = np.empty((2100, Nsim))
for i,E in enumerate(Es):
    for element in provider.model.elements: 
        element.material.young_modulus = E
    
    
    child_analyzer = analyzers.Linear(solver)
    parent_analyzer = analyzers.Static(provider, child_analyzer, linear_system)

   
    parent_analyzer.build_matrices()
    parent_analyzer.initialize()
    parent_analyzer.solve()
    
    U[:,i] = linear_system.solution
    
#TO DO write to file
#plt.hist(U[-2,:],10)
#plt.show()



