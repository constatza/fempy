# cython: language_level=3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 11:19:22 2019

@author: constatza
"""
import numpy as np
from time import time
import pickle, sys
import matplotlib.pyplot as plt
import smartplot as splt
import fem.core.providers as providers 

from fem.preprocessor import rectangular_mesh_model
from fem.problems import ProblemStructuralDynamic
from fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fem.solvers import SparseLUSolver#, SparseCholeskySolver
from fem.systems import LinearSystem

from fem.core.loads import InertiaLoad, TimeDependentLoad
from fem.core.entities import DOFtype

from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.elements import Quad4

sys.setrecursionlimit(10**6)
# plt.close('all')   
# =============================================================================
# INPUT
# =============================================================================
Nsim = 5 # 2500 in 71.7 min, seed=1
# np.random.seed(189)
output_suffix = 'InertiaLoadXY'

# =============================================================================
# LOADING
# =============================================================================

# DYNAMIC LOAD
total_time = 5
total_steps = 1000
reduced_step = 10
reduced_steps= np.arange(total_steps, step=reduced_step)
t = np.linspace(0, total_time, total_steps+1)
timestep = t[1]-t[0]

f0 = 1000
theta = np.random.rand(Nsim,) * np.pi/2

freq = np.random.rand(Nsim,)*50

phase = np.random.rand(Nsim,)*2*np.pi
F = f0 * np.sin(freq[:, None]*t + phase[:, None]) * np.exp(-t)
Fx = F * np.cos(theta[:, None]) 
Fy = F * np.sin(theta[:, None])


# MATERIAL PROPERTIES
Emean = 30
poisson_ratio = .2
thickness = 100
mass_density = 2.5e-9

# STOCHASTIC E FILES
stochastic_path = r"C:\Users\constatza\Documents\thesis\fempy\examples\stochastic_Young_Modulus\stochastic_E.npy"
Estochastic = np.load(stochastic_path, mmap_mode='r')
Estochastic = Estochastic[-Nsim:, :]
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
numelX = 30
numelY = 50
boundX = [0, 3000]
boundY = [0, 5000]

model = rectangular_mesh_model(boundX, boundY, 
                               numelX, numelY, quad)

# ASSIGN TIME DEPENDENT LOADS
Iload1 = InertiaLoad(time_history=F[0,:], DOF=DOFtype.X)
model.inertia_loads.append(Iload1)
model.inertia_loads.append(Iload1)
node_number = -10
# Iload1 = TimeDependentLoad(time_history=F[0,:], 
#                            node=model.nodes[node_number], 
#                            DOF=DOFtype.X)
# model.time_dependent_loads.append(Iload1)

# CONSTRAIN BASE DOFS
for node in model.nodes[:numelX+1]:
    node.constraints = [DOFtype.X, DOFtype.Y]
    
model.connect_data_structures()
damping_coeffs = [0.05, 0.05]
damping_provider = providers.RayleighDampingMatrixProvider(coeffs=damping_coeffs)
# =============================================================================
# BUILD ANALYZER
# =============================================================================
linear_system = LinearSystem(model.forces)
solver = SparseLUSolver(linear_system)

dynamic = ProblemStructuralDynamic(model, damping_provider=damping_provider)
dynamic.stiffness_provider = providers.ElementMaterialOnlyStiffnessProvider()
dynamic.change_mass = False
dynamic.global_matrix_provider = providers.GlobalMatrixProvider
child_analyzer = Linear(solver)
newmark = NewmarkDynamicAnalyzer(model=model, 
                                solver=solver, 
                                provider= dynamic, 
                                child_analyzer=child_analyzer, 
                                timestep=timestep, 
                                total_time=total_time)

displacements = np.empty((Nsim, model.total_DOFs, total_steps//reduced_step))
start = time()

for case in range(Nsim):
    print("Case {:d}".format(case))
    counter = -1
    seismic_loadX = InertiaLoad(time_history=Fx[case, :], DOF=DOFtype.X)#, node=model.nodes[node_number])
    seismic_loadY = InertiaLoad(time_history=Fy[case, :], DOF=DOFtype.Y)
    model.inertia_loads[0] = seismic_loadX
    model.inertia_loads[1] = seismic_loadY
    for width in range(numelX):
        for height in range(numelY):
            #slicing through elements list the geometry rectangle grid is columnwise
            counter += 1
            element = model.elements[counter] 
            element.material = ElasticMaterial2D(stress_state=StressState2D.plain_stress,
                                                 poisson_ratio=poisson_ratio,
                                                 young_modulus=Estochastic[case,
                                                                           height],
                                                 mass_density=mass_density)
    newmark.initialize()
    newmark.solve()
            
    displacements[case, :,:] = newmark.displacements[:, 1::reduced_step]

end = time()
print("Finished in {:.2f} min".format(end/60 - start/60) )




import fem.postprocessor as post

fig, ax = plt.subplots()
ax.set_aspect('equal')
i=20
ax.set_title('Displacements, t={:d}'.format(i))
model = post.assign_element_displacements(newmark.displacements[:,i], model)
ax = post.draw_deformed_shape(ax=ax, elements=model.elements, color='g', scale=200)



# =============================================================================
# SAVE
# =============================================================================
#
sformat = lambda x, suffix, ext: '_'.join((x, suffix)) + ext

with open(sformat('Problem', output_suffix, '.pickle'), 'wb') as file:
    pickle.dump(dynamic, file)

if case>=40:
    np.save(sformat('Displacements', output_suffix, '.npy'), displacements)
    np.save(sformat('Forces', output_suffix, '.npy'), F)

save_dict = {'Fx' : Fx,
             'Fy' : Fy,
             'f0' : f0,
             'phase' : phase,
             'frequency' : freq,
             'timeline' : t,
             'total_time' : total_time,
             'timestep' : timestep,
             'node_number' : node_number,
             
             'damping_coeffs' : damping_coeffs,
             'poisson_ratio' : poisson_ratio,
             'mass_density' : mass_density,
             'thickness' : thickness,
             
             'numelX' : numelX,
             'numelY' : numelY,
             'boundX' : boundX,
             'boundY' : boundY,
             'Nsim' : Nsim,
             
             'displacements_last_sim': newmark.displacements}


np.savez(sformat('FullOrder', output_suffix, '.npz'), **save_dict)

# =============================================================================
# PLOTS
# =============================================================================
displacements = newmark.displacements
timeline = range(displacements.shape[1])*timestep
# timeline = timeline[reduced_steps]
velocities = np.gradient(displacements, timestep, axis=1)
accelerations = np.gradient(velocities, timestep, axis=1)
node = 500
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

splt.gridplot(axes.ravel(), data)