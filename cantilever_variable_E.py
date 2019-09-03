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
numelX = 20
numelY = 50
boundX = [0, 2000]
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
"""
Nsim = 10000
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

"""
Displacements
"""
U = np.empty((2100, Nsim))
t1 = time.time()
for i,E in enumerate(Es):
    for element in provider.model.elements: 
        element.material.young_modulus = E
    
    
    child_analyzer = analyzers.Linear(solver)
    parent_analyzer = analyzers.Static(provider, child_analyzer, linear_system)

   
    parent_analyzer.build_matrices()
    parent_analyzer.initialize()
    parent_analyzer.solve()
    
    U[:,i] = linear_system.solution



print("Elapsed time = {:.3f} sec".format(time.time() - t1))
with open(__file__.split('.')[0] +"_output_U.csv", 'ab') as file:
    np.savetxt(file, U, delimiter=",", fmt="%.6f")

#TO DO write to file
#plt.hist(U[-2,:],10)
#plt.show()



