# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:03:04 2019

@author: constatza
"""
import pickle, gc

import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.mplot3d import Axes3D

import smartplot
import fem.core.providers as providers
import mathematics.manilearn as ml
import mathematics.stochastic as st
from fem.preprocessor import rectangular_mesh_model
from fem.problems import ProblemStructuralDynamic
from fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fem.solvers import CholeskySolver, SparseLUSolver
from fem.systems import LinearSystem
from fem.core.loads import InertiaLoad, TimeDependentLoad
from fem.core.entities import DOFtype
from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.elements import Quad4


plt.close('all')

# =============================================================================
# INPUT
# =============================================================================

Ntrain = 10
epsilon = 20
alpha = 0
numeigs = 5
diff_time = 1
use_pca = True
input_suffix = 'InertiaLoadXY'



# =============================================================================
# LOAD DATA FROM MONTE CARLO
# =============================================================================
# displacements
sformat = lambda x, suffix, ext: '_'.join((x, suffix)) + ext

stochastic_path = r"C:\Users\constatza\Documents\thesis\fempy\examples\stochastic_Young_Modulus\stochastic_E.npy"
E = np.load(stochastic_path, mmap_mode='r')
U = np.load(sformat('Displacements', input_suffix, '.npy'), mmap_mode='r')
# F = np.load(sformat('Forces', input_suffix, '.npy'))

Utrain = U[ :Ntrain, :, 1:10]
Utrain = np.concatenate(np.split(Utrain, Ntrain, axis=0), axis=2).squeeze()
# Utrain = st.zscore(Utrain, axis=2)
# Utrain = np.nan_to_num(Utrain)


FullOrder = np.load(sformat('FullOrder', input_suffix, '.npz'))

Fx = FullOrder['Fx']
Fy = FullOrder['Fy']
f0 = FullOrder['f0']
freq = FullOrder['frequency']
phase = FullOrder['phase'] 
timeline = FullOrder['timeline'][:-1]
timestep = FullOrder['timestep']
total_time =  FullOrder['total_time']
node_number = FullOrder['node_number']


damping_coeffs = FullOrder['damping_coeffs']
poisson_ratio = FullOrder['poisson_ratio']
mass_density = FullOrder['mass_density']
thickness = FullOrder['thickness']

numelX = FullOrder['numelX']
numelY = FullOrder['numelY'] 
boundX = FullOrder['boundX'] 
boundY = FullOrder['boundY'] 
Nsim = FullOrder['Nsim']

Ntest = FullOrder['Nsim']
Etest = E[-Ntest:, :]
# Ftest = F[-Ntest:, :]
# color = F[:Ntrain, :1000:10].reshape(-1)
# color = Utrain[-2, :]




del(U, E)
gc.collect()


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
pca_map.matrix = pca.reduced_coordinates

if use_pca:
    map_used = pca_map
else:
    map_used = diff_map
# =============================================================================
# MODEL CREATION
# =============================================================================

with open(sformat('Problem', input_suffix, '.pickle'), 'rb') as file:
    dynamic = pickle.load(file)
    model = dynamic.model


# ASSIGN TIME DEPENDENT LOADS

    
# model.connect_data_structures()
# damping_provider = providers.RayleighDampingMatrixProvider(coeffs=damping_coeffs)
# =============================================================================
# BUILD ANALYZER
# =============================================================================
linear_system = LinearSystem(model.forces)
solver = SparseLUSolver(linear_system)
dynamic.global_matrix_provider = providers.ReducedGlobalMatrixProvider(map_used)
dynamic.change_mass = False
child_analyzer = Linear(solver)
newmark = NewmarkDynamicAnalyzer(model=model, 
                                solver=solver, 
                                provider= dynamic, 
                                child_analyzer=child_analyzer, 
                                timestep=timestep, 
                                total_time=total_time)

# =============================================================================
# ANALYSES
# =============================================================================
# w = np.array([30.23187974]) #72
# phase = np.array([5.34745553])
P = f0 * np.sin(freq[:, None]*timeline + phase[:, None])
start = time()
for case in range(Ntest):
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
                                                  young_modulus=Etest[case, height],
                                                  mass_density=mass_density)
            
    newmark.initialize()
    newmark.solve()

end = time()
print("Finished in {:.2f} min".format(end/60 - start/60) )
# =============================================================================
# SAMPLING
# =============================================================================
plt.figure()

dof = 5
Ured = map_used.direct_transform_vector(newmark.displacements)
plt.plot(timeline, Ured[dof,:])
Ufull = FullOrder['displacements_last_sim']
plt.plot(timeline, Ufull[dof,:])
# plt.plot(timeline[::10], Utrain[-2, case*100:(case+1)*100])
# plt.plot(timeline, Ftest[case, :-1])

# pca r2 = 0.0431346999487219
# =============================================================================
# PLOTS
# =============================================================================

smartplot.paper_style()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# e = np.logspace(-2, 2, num=10)
# plt.loglog(e, dmaps.kernel_sums_per_epsilon(Utrain, epsilon=e))

fig, ax1 = plt.subplots()
smartplot.plot_eigenvalues(dmaps.eigenvalues,
                            ax=ax1,
                            log=True,
                            marker='+',
                            label='DMAPS')
smartplot.plot_eigenvalues(pca.eigenvalues/np.max(pca.eigenvalues),
                            ax=ax1,
                            log=True,
                            marker='x', 
                            label='PCA')
ax1.legend()


howmany = 4 
howmany = howmany * (numeigs > howmany) + (numeigs<= howmany) * numeigs
length = 1000

bw = 0.1
size = 3
dmaps_frame = pd.DataFrame(dmaps.znormed_eigenvectors[:howmany, :].T)
g = sns.PairGrid(dmaps_frame)
g.map_upper(plt.scatter, marker='.', s=size)
g.map_diag(sns.kdeplot, bw=5*bw)
g.map_lower(sns.kdeplot, bw=5*bw)
fig2 = g.fig
fig2.suptitle('DMAPS Normalized Eigenvectors')


pca_frame = pd.DataFrame(pca.znormed_eigenvectors[:howmany, :].T)
g = sns.PairGrid(pca_frame)
g.map_upper(plt.scatter, marker='.', s=size)
g.map_diag(sns.kdeplot, bw=bw)
g.map_lower(sns.kdeplot, bw=1.5*bw)
fig3 = g.fig
fig3.suptitle('PCA Normalized Eigenvectors')

# psi = [0,1,2]
# # set up a figure twice as wide as it is tall
# fig4 = plt.figure(figsize=plt.figaspect(0.3))
# ax41 = fig4.add_subplot(1, 3, 1, projection='3d')
# ax42 = fig4.add_subplot(1, 3, 2, projection='3d')
# ax43 = fig4.add_subplot(1, 3, 3, projection='3d')
# fig4.suptitle('Normalized Eigenvectors')
# smartplot.plot_eigenvectors(dmaps.znormed_eigenvectors[psi,:length],
#                             ax=ax42,
#                             title='DMAPS',
#                             psi=psi,
#                             # c=color[:length],
#                             marker='.')
# smartplot.plot_eigenvectors(pca.znormed_eigenvectors[psi,:length],
#                             ax=ax43,
#                             title='PCA',
#                             psi=psi,
#                             # c=color[:length],
#                             marker='.')

# smartplot.plot_eigenvectors(Utrain[psi,:length],
#                             ax=ax41, 
#                             title='Data',
#                             psi=psi,
#                             # c=color[:length],
#                             marker='.')


