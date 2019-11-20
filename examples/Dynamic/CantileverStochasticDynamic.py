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
from fempy.fem.core.providers import ElementMaterialOnlyStiffnessProvider, RayleighDampingMatrixProvider
from fempy.fem.core.materials import ElasticMaterial2D, StressState2D
from fempy.fem.core.elements import Quad4


# =============================================================================
# INPUT
# =============================================================================

Nsim = 10

# DYNAMIC LOAD

t = np.linspace(0, .1, 200)
timestep = t[1]-t[0]
total_time = 2*t[-1] 
f0 = 100
T0 = .5
T1 = .05
F = f0 * np.sin(2*np.pi*t/T1)

# MATERIAL PROPERTIES
Emean = 210
poisson_ratio = .3
thickness = 100
mass_density = 7.5e-9

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
damping_provider = RayleighDampingMatrixProvider(coeffs=[0.05, 0.05])
# =============================================================================
# BUILD ANALYZER
# =============================================================================
linear_system = LinearSystem(model.forces)
solver = CholeskySolver(linear_system)

provider = ProblemStructuralDynamic(model, damping_provider=None)
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
    
node = 1050
ux = parent_analyzer.displacements[:, 2*node-2]
uy = parent_analyzer.displacements[:, 2*node-1]
vx = parent_analyzer.velocities[:, 2*node-2]
vy = parent_analyzer.velocities[:, 2*node-1]
ax = parent_analyzer.accelerations[:, 2*node-2]
ay = parent_analyzer.accelerations[:, 2*node-1]
timeline = range(len(ux))*timestep

# =============================================================================
# PLOTS
# =============================================================================
#plt.close('all')
fig = plt.figure()
ax1 = fig.add_subplot(111)
splt.plot23d(ux, vx, ax=ax1, title='Phase Space')

fig, axes = plt.subplots(4, 1, sharex=True )

data = ((t, F),
        (timeline, ux, uy),
        (timeline, vx, vy),
        (timeline, ax, ay))


splt.gridplot(axes.ravel(), data)