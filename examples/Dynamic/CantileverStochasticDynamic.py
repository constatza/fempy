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

plt.close('all')   
# =============================================================================
# INPUT
# =============================================================================

Nsim = 10

# DYNAMIC LOAD

t = np.linspace(0, .5, 500)
timestep = t[1]-t[0]
total_time = 2*t[-1] 
f0 = 100
T0 = .5
T1 = .2
F = f0 * np.cos(2*np.pi*t/T1)

# MATERIAL PROPERTIES
Emean = 30
poisson_ratio = .3
thickness = 100
mass_density = 2.5e-9

# CANTILEVER SIZES
numelX = 20
numelY = 50
boundX = [0, 2000]
boundY = [0, 5000]

# STOCHASTIC E FILES
stochastic_path = r"C:\Users\constatza\Documents\thesis\fempy\examples\stochastic_Young_Modulus\stochastic_E.npy"
Estochastic = np.load(stochastic_path)
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
damping_provider = RayleighDampingMatrixProvider(coeffs=[0.001, 0.001])
# =============================================================================
# BUILD ANALYZER
# =============================================================================
linear_system = LinearSystem(model.forces)
solver = CholeskySolver(linear_system)

provider = ProblemStructuralDynamic(model, damping_provider=damping_provider)
provider.change_stiffness = True
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


for case in range(1):
    counter = -1
    
    for width in range(numelX):
        for height in range(numelY):
            #slicing through elements list the geometry rectangle grid is columnwise
            counter += 1
            element = model.elements[counter] 
            element.material.young_modulus = Estochastic[-1-case, height]
    print(element.material.young_modulus)        
    parent_analyzer.initialize()
    parent_analyzer.solve()
    
    node = 850
    
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

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    splt.plot23d(ux, vx, ax=ax1, title='Phase Space')
    
    fig, axes = plt.subplots(4, 1, sharex=True )
    
    data = ((t, F),
            (timeline, ux, uy),
            (timeline, vx, vy),
            (timeline, ax, ay))


    splt.gridplot(axes.ravel(), data)