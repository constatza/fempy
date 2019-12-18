# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:03:04 2019

@author: constatza
"""

import numpy as np
from time import time

import matplotlib.pyplot as plt
import smartplot

from fem.preprocessor import rectangular_mesh_model
from fem.problems import ProblemStructuralDynamic
from fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fem.solvers import CholeskySolver
from fem.systems import LinearSystem

from fem.core.loads import InertiaLoad
from fem.core.entities import DOFtype
from fem.core.providers import ElementMaterialOnlyStiffnessProvider, RayleighDampingMatrixProvider
from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.elements import Quad4

import mathematics.manilearn as ml
import mathematics.stochastic as st
import seaborn as sns
import pandas as pd 
plt.close('all')

Nsim = 50
epsilon = 8
alpha = 0
numeigs = 10
diff_time = 1


# =============================================================================
# LOAD DATA FROM MONTE CARLO
# =============================================================================
# displacements
U = np.load('U1_N2500.npy', mmap_mode='r')
F = np.load('F1_N2500.npy')
freq = np.load('Freq1.npy')
phase = np.load('Phase1.npy')
Ntrain = Nsim * U.shape[1]
Utrain = U[:, :, :Nsim]
Utrain1 = Utrain.transpose(2, 1, 0).reshape(Ntrain, -1).T
field = st.StochasticField(data=Utrain1, axis=1)
Utrain = Utrain1
color = F[:Nsim, :1000:10].reshape(-1)
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
linear_system = LinearSystem(model.forces)
solver = CholeskySolver(linear_system)

provider = ProblemStructuralDynamic(model, damping_provider=damping_provider)
provider.stiffness_provider = ElementMaterialOnlyStiffnessProvider()
child_analyzer = Linear(solver)
newmark = NewmarkDynamicAnalyzer(model=model, 
                                         solver=solver, 
                                         provider=provider, 
                                         child_analyzer=child_analyzer, 
                                         timestep=timestep, 
                                         total_time=total_time, 
                                         delta=1/2,
                                         alpha=1/4)


# =============================================================================
# SAMPLING
# =============================================================================


# =============================================================================
# DMAPS
# =============================================================================
dmaps = ml.DiffusionMap(Utrain, epsilon=epsilon, alpha=alpha)
dmaps.fit(numeigs=numeigs, t=diff_time) 
linear_dmaps = ml.LinearMap(domain=dmaps.reduced_coordinates, codomain=Utrain)
u_dm = linear_dmaps.direct_transform_vector(dmaps.reduced_coordinates) 

# =============================================================================
# PCA
# =============================================================================
pca = ml.PCA(Utrain)
pca.fit(numeigs=numeigs)
pca_map = ml.LinearMap(domain= pca.reduced_coordinates, codomain=Utrain)
u_pca = pca_map.direct_transform_vector(pca.reduced_coordinates)

# =============================================================================
# PLOTS
# =============================================================================

# fig = plt.figure()
# ax = fig.add_subplot(111)
# e = np.logspace(-2, 2, num=20)
# plt.loglog(e, dmaps.kernel_sums_per_epsilon(Utrain, e))

ax1 = smartplot.plot_eigenvalues(dmaps.eigenvalues, marker='+', label='DMAPS')
smartplot.plot_eigenvalues(pca.eigenvalues/np.max(pca.eigenvalues),ax=ax1, marker='x', label='PCA')
ax1.legend()
plt.grid()


howmany = 4
length = 5*100

bw = 0.1
dmaps_frame = pd.DataFrame(st.zscore(dmaps.eigenvectors[:howmany, :].T))
g = sns.PairGrid(dmaps_frame)
g.map_upper(plt.scatter, marker='.', s=1)
g.map_diag(sns.kdeplot, bw=bw)
g.map_lower(sns.kdeplot, bw=bw)
fig2 = g.fig
fig2.suptitle('DMAPS Normalized Eigenvectors')


pca_frame = pd.DataFrame(st.zscore(pca.eigenvectors[:howmany, :].T))
g = sns.PairGrid(pca_frame)
g.map_upper(plt.scatter, marker='.', s=1)
g.map_diag(sns.kdeplot, bw=bw)
g.map_lower(sns.kdeplot, bw=bw)
fig3 = g.fig
fig3.suptitle('PCA Normalized Eigenvectors')


fig4, axes4 = plt.subplots(1, 2)
fig4.suptitle('Normalized Eigenvectors')
smartplot.plot_eigenvectors(dmaps.eigenvectors[:howmany,length-100:length], ax=axes4[0], title='DMAPS', c=color[length-100:length], marker='.')
smartplot.plot_eigenvectors(pca.eigenvectors[:howmany,length-100:length], ax=axes4[1], title='PCA', c=color[length-100:length], marker='.')


fig5, axes5 = plt.subplots(1, 2)
fig5.suptitle('Correlation')
axes5[0].plot(color[length-100:length], dmaps.eigenvectors[1,length-100:length].T, marker='.', linestyle='')
axes5[1].plot(color[length-100:length], pca.reduced_coordinates[1,length-100:length].T, marker='.', linestyle='')