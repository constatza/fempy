# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""
import matplotlib.pyplot as plt
import numpy as np
import time 
import scipy.sparse.linalg as spla

import fem.preprocessor as pre
import fem.postprocessor as post
from fem.core.elements import Quad4
from fem.core.entities import DOFtype, Load
from fem.core.assemblers import ProblemStructural, ElementMaterialOnlyStiffnessProvider
from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.solvers import LinearSystem, SparseSolver, SparseLUSolver, ConjugateGradientSolver
import fem.analyzers as analyzers

plt.close()

"""
Units: kN, mm
"""

"""
Create rectangular mesh for the model.
"""
numelX = 14
numelY = 14*5
boundX = [0, 1000]
boundY = [0, 5000]
last_node = (numelX + 1) * (numelY + 1)

material = ElasticMaterial2D(stress_state=StressState2D.plain_stress,
                             young_modulus=None,
                             poisson_ratio=.3)
"""
Assign material
"""
quad = Quad4(material=material, thickness=100)

model = pre.rectangular_mesh_model(boundX, boundY, numelX, numelY, quad)
#pre.draw_mesh(model.elements, color='b')

for node in model.nodes[:numelX+1]:
    node.constraints = [DOFtype.X, DOFtype.Y]

"""
Assing Loads
units: kN
"""
load1 = Load(magnitude=100,
             node=model.nodes_dictionary[last_node-1], 
             DOF=DOFtype.X)


"""
Variable Young's modulus realization
units: kN/mm2 
"""
Nsim = 20
m = 30
v = (.2*m)**2
mu = np.log(m**2/(v + m**2)**.5)
sigma = (np.log(v/m**2 + 1))**.5
Es = np.random.lognormal(mean=mu, sigma=sigma, size=Nsim)   
#plt.hist(Es,30)
#plt.show()
"""
Model initialization
"""
model.loads.append(load1)             
model.connect_data_structures()
linear_system = LinearSystem(model.forces)
solver = SparseLUSolver(linear_system)
provider = ProblemStructural(model)
provider.stiffness_provider = ElementMaterialOnlyStiffnessProvider() 
child_analyzer = analyzers.Linear(solver)
parent_analyzer = analyzers.Static(provider, child_analyzer, linear_system)

import pickle
with open('analyzer.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(parent_analyzer, output, pickle.HIGHEST_PROTOCOL)

"""
Displacements
"""
U = np.empty((Nsim, 2100))
t1 = time.time()
for i,E in enumerate(Es):
    for element in parent_analyzer.provider.model.elements: 
        element.material.young_modulus = E
    
    
    

   
    parent_analyzer.build_matrices()
    parent_analyzer.initialize()
    parent_analyzer.solve()
    
    U[i, :] = linear_system.solution

print("Elapsed time = {:.3f} sec".format(time.time() - t1))
print("-------------------------")

model = post.assign_element_displacements(U[1,:], model)
ax = post.draw_deformed_shape(elements=model.elements, scale=20, color='g')
ax.set_aspect('equal', adjustable='box')
plt.draw()

#np.save('variable_E_displacements', U)






