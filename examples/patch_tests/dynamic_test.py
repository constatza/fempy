# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:04:34 2019

@author: constatza
"""
import numpy as np
import matplotlib.pyplot as plt

from fempy.fem.problems import ProblemStructuralDynamic
from fempy.fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fempy.fem.solvers import CholeskySolver
from fempy.fem.systems import LinearSystem

from fempy.fem.core.loads import Load, TimeDependentLoad
from fempy.fem.core.entities import Model, Node, DOFtype
from fempy.fem.core.providers import ElementMaterialOnlyStiffnessProvider
from fempy.fem.core.materials import ElasticMaterial2D, StressState2D
from fempy.fem.core.elements import Quad4



model = Model()

model.nodes_dictionary[1] = Node(ID=1, X=0.0, Y=0.0, Z=0.0)
model.nodes_dictionary[2] = Node(ID=2, X=10.0, Y=0.0, Z=0.0)
model.nodes_dictionary[3] = Node(ID=3, X=10.0, Y=10.0, Z=0.0)
model.nodes_dictionary[4] = Node(ID=4, X=0.0, Y=10.0, Z=0.0)

model.nodes_dictionary[1].constraints = [DOFtype.X, DOFtype.Y]
model.nodes_dictionary[4].constraints = [DOFtype.X, DOFtype.Y]

load1 = Load(magnitude=1000, node=model.nodes_dictionary[2], DOF=DOFtype.X)
load2 = Load(magnitude=1000, node=model.nodes_dictionary[3], DOF=DOFtype.X)
model.loads.append(load1)
model.loads.append(load2)
t = np.linspace(0, 1)
timestep = t[1] -t[0]
F0 = 1000.0
history = F0 * np.cos(2*3.14*t) * np.random.rand(t.shape[0])
hload = TimeDependentLoad(time_history=history, 
                          node = model.nodes_dictionary[2], 
                          DOF=DOFtype.X)
                          
model.time_dependent_loads.append(hload)
material = ElasticMaterial2D(stress_state=StressState2D.plain_stress,
                             young_modulus=21000,
                             poisson_ratio=0.3779,
                             mass_density = 7.85)

quad = Quad4(material=material, thickness=1)
element1 = Quad4(ID=1, material=material, element_type=quad, thickness=1)
for i in range(1,5):
    element1.add_node(model.nodes_dictionary[i])			

model.elements_dictionary[1] = element1

model.connect_data_structures()

linear_system = LinearSystem(model.forces)
solver = CholeskySolver(linear_system)
    
provider = ProblemStructuralDynamic(model)
provider.stiffness_provider = ElementMaterialOnlyStiffnessProvider()
child_analyzer = Linear(solver)
parent_analyzer = NewmarkDynamicAnalyzer(model, 
                                         solver, 
                                         provider, 
                                         child_analyzer, 
                                         timestep=timestep, 
                                         total_time=1000, 
                                         delta=1/2,
                                         alpha=1/4)


for i in range(1):
    parent_analyzer.build_matrices()
    parent_analyzer.initialize()
    parent_analyzer.solve()

u = parent_analyzer.displacement[:, 2]
v = parent_analyzer.velocity[:, 2]
a = parent_analyzer.velocity[:, 2]

import matplotlib.animation as animation
plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot([], [])

def init(x,y):
    line1.axes.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    return line1,
    

def update(num, x, y, line1):
    line1.set_data(x[:num], y[:num])
    line1.axes.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
    return line1,

timeline = range(len(u))*timestep 
ani =animation.FuncAnimation(fig, update, len(u), fargs=[u, v, line1],
                  interval=timestep*1000)
    
plt.show()