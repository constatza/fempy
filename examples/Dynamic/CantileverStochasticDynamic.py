# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:19:22 2019

@author: constatza
"""

import numpy as np
import matplotlib.pyplot as plt
import fempy.smartplot as splt

from fempy.fem.preprocessor import rectangular_mesh_model
from fempy.fem.problems import ProblemStructuralDynamic
from fempy.fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fempy.fem.solvers import SparseLUSolver, CholeskySolver
from fempy.fem.systems import LinearSystem

from fempy.fem.core.loads import Load, TimeDependentLoad
from fempy.fem.core.entities import Model, Node, DOFtype
from fempy.fem.core.providers import ElementMaterialOnlyStiffnessProvider
from fempy.fem.core.materials import ElasticMaterial2D, StressState2D
from fempy.fem.core.elements import Quad4

plt.close('all')
# =============================================================================
# INPUT
# =============================================================================

Nsim = 10

# DYNAMIC LOAD

t = np.linspace(0, 5, 2000)
timestep = t[1]-t[0]
total_time = 8
f0 = 100
w = 10
F = f0 * np.sin(w*t)

# MATERIAL PROPERTIES
Emean = 30
poisson_ratio = .3
thickness = 100
mass_density = 7.8e-9

# CANTILEVER SIZES
numelX = 20
numelY = 50
boundX = [0, 2000]
boundY = [0, 5000]

# =============================================================================
# MODEL CREATION
# =============================================================================

# CREATE MATERIAL TYPE
material = ElasticMaterial2D(stress_state=StressState2D.plain_stress,
                             poisson_ratio=poisson_ratio,
                             young_modulus=Emean,
                             mass_density=mass_density)
# CREATE ELEMENT TYPE
quad = Quad4(material=material, thickness=thickness)

model = rectangular_mesh_model(boundX, boundY, 
                               numelX, numelY, quad)
# ASSIGN TIME DEPENDENT LOADS
last_node = (numelX + 1) * (numelY + 1)
hload1 = TimeDependentLoad(time_history=F, 
                          node = model.nodes_dictionary[last_node-1], 
                          DOF=DOFtype.X)
model.time_dependent_loads.append(hload1)

# CONSTRAIN BASE DOFS
for node in model.nodes[:numelX+1]:
    node.constraints = [DOFtype.X, DOFtype.Y]
    
model.connect_data_structures()

# =============================================================================
# BUILD ANALYZER
# =============================================================================
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
    parent_analyzer.build_matrices()
    parent_analyzer.initialize()
    parent_analyzer.solve()
    
node = 1030
ux = parent_analyzer.displacement[:, 2*node-2]
uy = parent_analyzer.displacement[:, 2*node-1]
vx = parent_analyzer.velocity[:, 2*node-2]
vy = parent_analyzer.velocity[:, 2*node-1]
ax = parent_analyzer.acceleration[:, 2*node-2]
ay = parent_analyzer.acceleration[:, 2*node-1]
timeline = range(len(ux))*timestep



fig = plt.figure()
ax1 = fig.add_subplot(111)
splt.plot23d(timeline, ux, ax=ax1, title='Velocities')

fig, axes = plt.subplots(3, 1)

data = ((timeline, ux, uy),
        (timeline, vx, vy),
        (timeline, ax, ay))


splt.gridplot(axes.ravel(), data )