# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:03:04 2019

@author: constatza
"""

import numpy as np
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import smartplot
import fem.core.providers as providers

from fem.preprocessor import rectangular_mesh_model
from fem.problems import ProblemStructuralDynamic
from fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fem.solvers import CholeskySolver, SparseSolver
from fem.systems import LinearSystem

from fem.core.loads import InertiaLoad
from fem.core.entities import DOFtype

from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.elements import Quad4

import mathematics.manilearn as ml
import mathematics.stochastic as st
import seaborn as sns
import pandas as pd 
plt.close('all')

Ntrain = 60
Ntest = 10
epsilon = 10
alpha = 0
numeigs = 4
diff_time = 1


# =============================================================================
# LOAD DATA FROM MONTE CARLO
# =============================================================================
# displacements
U = np.load('U1_N2500.npy', mmap_mode='r')
F = np.load('F1_N2500.npy')
freq = np.load('Freq1.npy')
phase = np.load('Phase1.npy')
Ntrain_total = Ntrain * U.shape[1]
Utrain = U[:, :, :Ntrain]
Utrain1 = Utrain.transpose(2, 1, 0).reshape(Ntrain_total, -1).T
field = st.StochasticField(data=Utrain1, axis=1)
Utrain = Utrain1

stochastic_path = r"C:\Users\constatza\Documents\thesis\fempy\examples\stochastic_Young_Modulus\stochastic_E.npy"
E = np.load(stochastic_path, mmap_mode='r')
Estochastic = E[Ntrain:Ntrain + Ntest, :]
color = F[:Ntrain, :1000:10].reshape(-1)
# =============================================================================
# DMAPS
# =============================================================================
dmaps = ml.DiffusionMap(Utrain, epsilon=epsilon, alpha=alpha)
dmaps.fit(numeigs=numeigs, t=diff_time) 
diff_map = ml.LinearMap(domain=dmaps.reduced_coordinates, codomain=Utrain)
u_dm = diff_map.direct_transform_vector(dmaps.reduced_coordinates) 

# =============================================================================
# PCA
# =============================================================================
pca = ml.PCA(Utrain)
pca.fit(numeigs=numeigs)
pca_map = ml.LinearMap(domain= pca.reduced_coordinates, codomain=Utrain)
u_pca = pca_map.direct_transform_vector(pca.reduced_coordinates)

# =============================================================================
# MODEL CREATION
# =============================================================================

# time data
total_time = 10
total_steps = 1000
reduced_step = 10
reduced_steps= np.arange(total_steps, step=reduced_step)
t = np.linspace(0, total_time, total_steps+1)
timestep = t[1]-t[0]

# MATERIAL PROPERTIES
Emean = 30
poisson_ratio = .2
thickness = 100
mass_density = 2.5e-9

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

Iload1 = InertiaLoad(time_history=F[0, :], DOF=DOFtype.X)
model.inertia_loads.append(Iload1)


# CONSTRAIN BASE DOFS
for node in model.nodes[:numelX+1]:
    node.constraints = [DOFtype.X, DOFtype.Y]
    
model.connect_data_structures()
damping_provider = providers.RayleighDampingMatrixProvider(coeffs=[0.1, 0.1])
# =============================================================================
# BUILD ANALYZER
# =============================================================================
linear_system = LinearSystem(model.forces)
solver = CholeskySolver(linear_system)

dynamic = ProblemStructuralDynamic(model, damping_provider=damping_provider)
dynamic.stiffness_provider = providers.ElementMaterialOnlyStiffnessProvider()
dynamic.global_matrix_provider = providers.ReducedGlobalMatrixProvider(diff_map)
child_analyzer = Linear(solver)
newmark = NewmarkDynamicAnalyzer(model=model, 
                                solver=solver, 
                                provider= dynamic, 
                                child_analyzer=child_analyzer, 
                                timestep=timestep, 
                                total_time=total_time, 
                                delta=1/2,
                                alpha=1/4)

# =============================================================================
# ANALYSES
# =============================================================================
start = time()
for case in range(Ntest):
    print("Case {:d}".format(case))
    counter = -1
    seismic_load = InertiaLoad(time_history=F[case, :], DOF=DOFtype.X)
    model.inertia_loads[0] = seismic_load
    for width in range(numelX):
        for height in range(numelY):
            #slicing through elements list the geometry rectangle grid is columnwise
            counter += 1
            element = model.elements[counter] 
            element.material.young_modulus = Estochastic[case, height]
            
    newmark.initialize()
    newmark.solve()

end = time()
print("Finished in {:.2f} min".format(end/60 - start/60) )
# =============================================================================
# SAMPLING
# =============================================================================
plt.figure()


timeline = timestep* range(newmark.displacements.shape[1])
Ur = diff_map.matrix @ newmark.displacements
plt.plot(timeline, Ur[-2,:])
plt.plot(timeline[::1], F[case, :-1])


# =============================================================================
# PLOTS
# =============================================================================

# smartplot.paper_style()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# e = np.logspace(-2, 2, num=10)
# plt.loglog(e, dmaps.kernel_sums_per_epsilon(Utrain, e))

# fig, ax1 = plt.subplots()
# smartplot.plot_eigenvalues(dmaps.eigenvalues,
#                            ax=ax1,
#                            marker='+',
#                            label='DMAPS')
# smartplot.plot_eigenvalues(pca.eigenvalues/np.max(pca.eigenvalues),
#                            ax=ax1, 
#                            marker='x', 
#                            label='PCA')
# ax1.legend()


# howmany = 4 
# howmany = howmany * (numeigs > howmany) + (numeigs<= howmany) * numeigs
# length = 100

# bw = 0.1
# size = 3
# dmaps_frame = pd.DataFrame(st.zscore(dmaps.eigenvectors[:howmany, :].T))
# g = sns.PairGrid(dmaps_frame)
# g.map_upper(plt.scatter, marker='.', s=size)
# g.map_diag(sns.kdeplot, bw=bw)
# g.map_lower(sns.kdeplot, bw=1.5*bw)
# fig2 = g.fig
# fig2.suptitle('DMAPS Normalized Eigenvectors')


# pca_frame = pd.DataFrame(st.zscore(pca.eigenvectors[:howmany, :].T))
# g = sns.PairGrid(pca_frame)
# g.map_upper(plt.scatter, marker='.', s=size)
# g.map_diag(sns.kdeplot, bw=bw)
# g.map_lower(sns.kdeplot, bw=1.5*bw)
# fig3 = g.fig
# fig3.suptitle('PCA Normalized Eigenvectors')

# psi = [0,1,2]
# # set up a figure twice as wide as it is tall
# fig4 = plt.figure(figsize=plt.figaspect(0.3))
# ax41 = fig4.add_subplot(1, 3, 1, projection='3d')
# ax42 = fig4.add_subplot(1, 3, 2, projection='3d')
# ax43 = fig4.add_subplot(1, 3, 3, projection='3d')
# fig4.suptitle('Normalized Eigenvectors')
# smartplot.plot_eigenvectors(dmaps.eigenvectors[psi,:length],
#                             ax=ax42,
#                             title='DMAPS',
#                             psi=psi,
#                             c=color[:length],
#                             marker='.')
# smartplot.plot_eigenvectors(pca.eigenvectors[psi,:length],
#                             ax=ax43,
#                             title='PCA',
#                             psi=psi,
#                             c=color[:length],
#                             marker='.')

# smartplot.plot_eigenvectors(Utrain[psi,:length],
#                             ax=ax41, 
#                             title='Data',
#                             psi=psi,
#                             c=color[:length],
#                             marker='.')


# fig5, axes5 = plt.subplots(1, 2)
# fig5.suptitle('Correlation')
# axes5[0].plot(color[:length], dmaps.eigenvectors[1,:length].T, marker='.', linestyle='')
# axes5[1].plot(color[:length], pca.reduced_coordinates[1,:length].T, marker='.', linestyle='')