# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import time 


import fempy.fem.preprocessor as pre
import fempy.fem.postprocessor as post
from fempy.fem.core.elements import Quad4
from fempy.fem.core.entities import DOFtype, Load
from fempy.fem.core.assemblers import ProblemStructural, ElementMaterialOnlyStiffnessProvider
from fempy.fem.core.materials import ElasticMaterial2D, StressState2D
from fempy.fem.core.solvers import LinearSystem, SparseLUSolver
import fempy.fem.analyzers as analyzers

sys.setrecursionlimit(100000)
plt.close()



# =============================================================================
# MODEL PROPERTIES
# Units: kN, mm
# =============================================================================

numelX = 20
numelY = 50
boundX = [0, 2000]
boundY = [0, 5000]

Nsim = 10
Emean = 30
poisson_ratio = .3
thickness = 100

"""Assign material"""
material = ElasticMaterial2D(stress_state=StressState2D.plain_stress,
                             young_modulus=None,
                             poisson_ratio=poisson_ratio)

quad = Quad4(material=material, thickness=thickness)

model = pre.rectangular_mesh_model(boundX, boundY, numelX, numelY, quad)

"""Constrain base DOFs"""
for node in model.nodes[:numelX+1]:
    node.constraints = [DOFtype.X, DOFtype.Y]

"""
Assing Loads
units: kN
"""
last_node = (numelX + 1) * (numelY + 1)
load1 = Load(magnitude=100,
             node=model.nodes_dictionary[last_node-1], 
             DOF=DOFtype.X)


"""
Variable Young's modulus realization
units: kN/mm2 
"""

m = Emean
v = (.2*m)**2
mu = np.log(m**2/(v + m**2)**.5)
sigma = (np.log(v/m**2 + 1))**.5
Es = np.random.lognormal(mean=mu, sigma=sigma, size=Nsim)   
np.save("young_modulus_{:d}x{:d}".format(numelX, numelY), Es)
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
ax = post.draw_deformed_shape(elements=model.elements, scale=200, color='g')
ax.set_aspect('equal', adjustable='box')
plt.draw()

#np.save('variable_E_displacements_{:d}x{:d}'.format(numelX, numelY), U)






