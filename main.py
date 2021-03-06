# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 21:12:36 2019

@author: constatza
"""
import matplotlib.pyplot as plt

from fem.core.entities import Model, Node, DOFtype, Load
from fem.core.assemblers import ProblemStructural, ElementMaterialOnlyStiffnessProvider
from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.elements import Quad4
from fem.core.solvers import LinearSystem, CholeskySolver
import fem.analyzers as analyzers

model = Model()

model.nodes_dictionary[1] = Node(ID=1, X=0.0, Y=0.0, Z=0.0)
model.nodes_dictionary[2] = Node(ID=2, X=10.0, Y=0.0, Z=0.0)
model.nodes_dictionary[3] = Node(ID=3, X=10.0, Y=10.0, Z=0.0)
model.nodes_dictionary[4] = Node(ID=4, X=0.0, Y=10.0, Z=0.0)

model.nodes_dictionary[1].constraints = [DOFtype.X, DOFtype.Y]
model.nodes_dictionary[4].constraints = [DOFtype.X, DOFtype.Y]

load1 = Load(magnitude=500, node=model.nodes_dictionary[2], DOF=DOFtype.X)
load2 = Load(magnitude=500, node=model.nodes_dictionary[3], DOF=DOFtype.X)
model.loads.append(load1)
model.loads.append(load2)

material = ElasticMaterial2D(stress_state=StressState2D.plain_stress,
                             young_modulus=3.76,
                             poisson_ratio=0.3779)

quad = Quad4(material=material, thickness=1)
element1 = Quad4(ID=1, material=material, element_type=quad, thickness=1)
for i in range(1,5):
    element1.add_node(model.nodes_dictionary[i])			

model.elements_dictionary[1] = element1

model.connect_data_structures()

linear_system = LinearSystem(model.forces)
solver = CholeskySolver(linear_system)
    
provider = ProblemStructural(model)
provider.stiffness_provider = ElementMaterialOnlyStiffnessProvider()
child_analyzer = analyzers.Linear(solver)
parent_analyzer = analyzers.Static(provider, child_analyzer, linear_system)


for i in range(20000):
    
    parent_analyzer.build_matrices()
    parent_analyzer.initialize()
    parent_analyzer.solve()

u = linear_system.solution

#plt.figure()
#plt.plot(u, linestyle=' ', marker='.')
#plt.show()