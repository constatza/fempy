# cython: language_level=3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:19:22 2019

@author: constatza
"""
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from time import time

import matplotlib.pyplot as plt
import smartplot as splt

from fem.preprocessor import rectangular_mesh_model
from fem.problems import ProblemStructuralDynamic
from fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fem.solvers import CholeskySolver
from fem.systems import LinearSystem

from fem.core.loads import TimeDependentLoad, InertiaLoad
from fem.core.entities import DOFtype
from fem.core.providers import ElementMaterialOnlyStiffnessProvider, RayleighDampingMatrixProvider
from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.elements import Quad4

plt.close('all')   
# =============================================================================
# INPUT
# =============================================================================

Nsim = 10

# DYNAMIC LOAD

t = np.linspace(0, 2, 400)
timestep = t[1]-t[0]
total_time = 2*t[-1] 
f0 = 100
T0 = .5
T1 = .2
F = f0 * np.exp(-(t-t[-1]/2)**2/T0/T0)*np.sin(2*np.pi*t/T1)
#F = f0 *np.ones(t.shape)
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
#num_last_node = (numelX + 1) * (numelY + 1)
#last_node=model.nodes_dictionary[last_node-1],
Iload1 = InertiaLoad(time_history=F, DOF=DOFtype.X)
#Iload2 = InertiaLoad(time_history=F,DOF=DOFtype.Y)
model.inertia_loads.append(Iload1)
#model.inertia_loads.append(Iload2)

# CONSTRAIN BASE DOFS
for node in model.nodes[:numelX+1]:
    node.constraints = [DOFtype.X, DOFtype.Y]
    
model.connect_data_structures()
damping_provider = RayleighDampingMatrixProvider(coeffs=[0.1, 0.1])
# =============================================================================
# BUILD ANALYZER
# =============================================================================
linear_system = LinearSystem(model.forces)
solver = CholeskySolver(linear_system)

provider = ProblemStructuralDynamic(model, damping_provider=damping_provider)
provider.stiffness_provider = ElementMaterialOnlyStiffnessProvider()
child_analyzer = Linear(solver)
parent_analyzer = NewmarkDynamicAnalyzer(model=model, 
                                         solver=solver, 
                                         provider=provider, 
                                         child_analyzer=child_analyzer, 
                                         timestep=timestep, 
                                         total_time=total_time, 
                                         delta=1/2,
                                         alpha=1/4)

start = time()

#for case in range(2):
#    counter = -1
#    
#    for width in range(numelX):
#        for height in range(numelY):
#            #slicing through elements list the geometry rectangle grid is columnwise
#            counter += 1
#            element = model.elements[counter] 
#            element.material.young_modulus = Estochastic[case, height]
#    print(element.material.young_modulus)        
#    parent_analyzer.initialize()
#    parent_analyzer.solve()

import fem.analysis as analysis
analysis.material_monte_carlo(parent_analyzer, Estochastic[:2,:], numelX, numelY)

displacements = parent_analyzer.displacements
timeline = range(displacements.shape[0])*timestep
velocities = np.gradient(displacements, timestep, axis=0)
accelerations = np.gradient(velocities, timestep, axis=0)
node = 1020
ux = displacements[:, 2*node-2]
uy = displacements[:, 2*node-1]
vx = velocities[:, 2*node-2]
vy = velocities[:, 2*node-1]
ax = accelerations[:, 2*node-2]
ay = accelerations[:, 2*node-1]

end = time()

print("Finished in {:.2f}".format(end - start) )
# =============================================================================
# PLOTS
# =============================================================================

fig = plt.figure()
ax1 = fig.add_subplot(111)
splt.plot23d(ux, uy, ax=ax1, title='Phase Space')

fig, axes = plt.subplots(4, 1, sharex=True )

data = ((t, F),
        (timeline, ux, uy),
        (timeline, vx, vy),
        (timeline, ax, ay))


splt.gridplot(axes.ravel(), data)