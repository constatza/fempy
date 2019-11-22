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
model.nodes_dictionary[2] = Node(ID=2, X=10e3, Y=0.0, Z=0.0)
model.nodes_dictionary[3] = Node(ID=3, X=10e3, Y=10e3, Z=0.0)
model.nodes_dictionary[4] = Node(ID=4, X=0.0, Y=10e3, Z=0.0)

model.nodes_dictionary[1].constraints = [DOFtype.X, DOFtype.Y]
model.nodes_dictionary[4].constraints = [DOFtype.X, DOFtype.Y]

load1 = Load(magnitude=1000, node=model.nodes_dictionary[2], DOF=DOFtype.X)
load2 = Load(magnitude=1000, node=model.nodes_dictionary[3], DOF=DOFtype.X)
#model.loads.append(load1)
#model.loads.append(load2)
t = np.linspace(0, 1, 200)
timestep = t[1] -t[0]
total_time = 5
F0 = 1000.0
history = F0* np.cos(20*3.14*t)
hload1 = TimeDependentLoad(time_history=history, 
                          node = model.nodes_dictionary[2], 
                          DOF=DOFtype.X)
hload2 = TimeDependentLoad(time_history=history, 
                          node = model.nodes_dictionary[3], 
                          DOF=DOFtype.X)
                          
model.time_dependent_loads.append(hload1)
model.time_dependent_loads.append(hload2)
material = ElasticMaterial2D(stress_state=StressState2D.plain_stress,
                             young_modulus=210e0,
                             poisson_ratio=0.3779,
                             mass_density = 7.85e-9)

quad = Quad4(material=material)
element1 = Quad4(ID=1, material=material, element_type=quad, thickness=1e1)
for i in range(1, 5):
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
                                         total_time=total_time, 
                                         delta=1/2,
                                         alpha=1/4)


for i in range(1):
    
    parent_analyzer.initialize()
    parent_analyzer.solve()


node = 1
ux = parent_analyzer.displacements[:, 2*node-2]
uy = parent_analyzer.displacements[:, 2*node-1]
vx = parent_analyzer.velocities[:, 2*node-2]
vy = parent_analyzer.velocities[:, 2*node-1]
ax = parent_analyzer.accelerations[:, 2*node-2]
ay = parent_analyzer.accelerations[:, 2*node-1]
timeline = range(len(ux))*timestep

import fempy.smartplot as sp
plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(111)
sp.plot23d(timeline, ux, ax=ax1, title='Velocities')

fig, axes = plt.subplots(3, 1)

data = ((timeline, ux, uy),
        (timeline, vx, vy),
        (timeline, ax, ay))


sp.gridplot(axes.ravel(), data )


#import matplotlib.animation as animation
#plt.close('all')
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#line1, = ax1.plot([], [])
#line2, = ax1.plot([], [])
#
#
#def update2d(num, x, y, lines):
#    if not isinstance(x, list):
#        lines.set_data(x[:num], y[:num])
#        lines.axes.axis([np.min(x), np.max(x), np.min(y), np.max(y)])
#    else:
#        for i in range(len(lines)):
#            lines[i].set_data(x[i][:num], y[i][:num])
#        lines[i].axes.axis([np.min(x), np.max(x), np.min(y), np.max(y)])    
#        
#    return lines
#
# 
#ani =animation.FuncAnimation(fig, update2d, len(ux), fargs=[ax, ay, line1],
#                interval=timestep*1000)
#    
#plt.show()