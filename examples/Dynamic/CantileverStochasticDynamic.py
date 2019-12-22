# cython: language_level=3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:19:22 2019

@author: constatza
"""
import numpy as np
# import pyximport
# pyximport.install(setup_args={'include_dirs': np.get_include()})
from time import time

import matplotlib.pyplot as plt
import smartplot as splt

from fem.preprocessor import rectangular_mesh_model
from fem.problems import ProblemStructuralDynamic
from fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fem.solvers import CholeskySolver, SparseLUSolver#, SparseCholeskySolver
from fem.systems import LinearSystem

from fem.core.loads import InertiaLoad
from fem.core.entities import DOFtype
from fem.core.providers import ElementMaterialOnlyStiffnessProvider, RayleighDampingMatrixProvider
from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.elements import Quad4

plt.close('all')   
# =============================================================================
# INPUT
# =============================================================================
# 2500 in 71.7 min
Nsim = 2500
np.random.seed(1)
# DYNAMIC LOAD
total_time = 5
total_steps = 1000
reduced_step = 10
reduced_steps= np.arange(total_steps, step=reduced_step)
t = np.linspace(0, total_time, total_steps+1)
timestep = t[1]-t[0]

f0 = 1000
period = np.random.rand(Nsim,)*2 + .05
w = np.random.rand(Nsim,)*80
phase = np.random.rand(Nsim,)*2*np.pi
F = f0 * np.sin(w[:, None]*t + phase[:, None])

#F = f0 *np.ones(t.shape)
# MATERIAL PROPERTIES
Emean = 30
poisson_ratio = .2
thickness = 100
mass_density = 2.5e-9



# STOCHASTIC E FILES
stochastic_path = r"C:\Users\constatza\Documents\thesis\fempy\examples\stochastic_Young_Modulus\stochastic_E.npy"
Estochastic = np.load(stochastic_path, mmap_mode='r')
Estochastic = Estochastic[:Nsim, :]
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

# CANTILEVER SIZES
numelX = 20
numelY = 50
boundX = [0, 2000]
boundY = [0, 5000]

model = rectangular_mesh_model(boundX, boundY, 
                               numelX, numelY, quad)

# ASSIGN TIME DEPENDENT LOADS
Iload1 = InertiaLoad(time_history=F[0,:], DOF=DOFtype.X)
model.inertia_loads.append(Iload1)


# CONSTRAIN BASE DOFS
for node in model.nodes[:numelX+1]:
    node.constraints = [DOFtype.X, DOFtype.Y]
    
model.connect_data_structures()
damping_provider = RayleighDampingMatrixProvider(coeffs=[0.1, 0.1])
# =============================================================================
# BUILD ANALYZER
# =============================================================================
linear_system = Lineanear_system)

provider = ProblemStructurrSystem(model.forces)
solver = SparseLUSolver(lialDynamic(model, 
                                    damping_provider=damping_provider)
provider.change_mass = False
provider.stiffness_provider = ElementMaterialOnlyStiffnessProvider()
child_analyzer = Linear(solver)
newmark = NewmarkDynamicAnalyzer(model=model, 
                                         solver=solver, 
                                         provider=provider, 
                                         child_analyzer=child_analyzer, 
                                         timestep=timestep, 
                                         total_time=total_time, 
                                         )
displacements = np.empty((model.total_DOFs, total_steps//reduced_step, Nsim))
start = time()

for case in range(Nsim):
    print("Case {:d}".format(case))
    counter = -1
    seismic_load = InertiaLoad(time_history=F[case, :], DOF=DOFtype.X)
    model.inertia_loads[0] = seismic_load
    for width in range(numelX):
        for height in range(numelY):
            #slicing through elements list the geometry rectangle grid is columnwise
            counter += 1
            element = model.elements[counter] 
            element.material = ElasticMaterial2D(stress_state=StressState2D.plain_stress,
                                                 poisson_ratio=poisson_ratio,
                                                 young_modulus=Estochastic[case,height],
                                                 mass_density=mass_density)
    newmark.initialize()
    newmark.solve()
            


    displacements[:,:, case] = newmark.displacements[:, ::reduced_step]

end = time()
print("Finished in {:.2f} min".format(end/60 - start/60) )
# =============================================================================
# SAVE
# =============================================================================
# 
if Nsim>500:
    np.save('Displacements1.npy', displacements)
    np.save('Forces1.npy', F)
    np.save('Phase1.npy', phase)
    np.save('Freq1.npy', w)







# =============================================================================
# PLOTS
# =============================================================================
displacements = newmark.displacements
timeline = range(displacements.shape[1])*timestep
# timeline = timeline[reduced_steps]
velocities = np.gradient(displacements, timestep, axis=1)
accelerations = np.gradient(velocities, timestep, axis=1)
node = 1020
ux = displacements[2*node-2, :]
uy = displacements[2*node-1, :]
vx = velocities[2*node-2, :]
vy = velocities[2*node-1, :]
ax = accelerations[2*node-2, :]
ay = accelerations[2*node-1, :]
fig = plt.figure()
ax1 = fig.add_subplot(111)
splt.plot23d(ux, uy, ax=ax1, title='Phase Space')

fig, axes = plt.subplots(4, 1, sharex=True )

data = ((t, F[case, :]),
        (timeline, ux, uy),
        (timeline, vx, vy),
        (timeline, ax, ay))

np.save('u2038.npy', ux)
splt.gridplot(axes.ravel(), data)