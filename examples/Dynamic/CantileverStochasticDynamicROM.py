# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:03:04 2019

@author: constatza
"""
import pickle, gc
import os
import numpy as np
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.mplot3d import Axes3D

import fem.core.providers as providers
import mathematics.manilearn as ml
from fem.systems import LinearSystem
from fem.core.entities import DOFtype
from fem.core.materials import ElasticMaterial2D, StressState2D
from fem.core.loads import InertiaLoad#, TimeDependentLoad
from fem.analyzers import Linear, NewmarkDynamicAnalyzer
from fem.solvers import SparseLUSolver


# =============================================================================
# INPUT
# =============================================================================
# 2800 in 30.776 min
Ntrain = 60
epsilon = 20
alpha = 0
numeigs = 3
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

Utrain = U[ :Ntrain, :, 1::10]
Utrain = np.concatenate(np.split(Utrain, Ntrain, axis=0), axis=2).squeeze()
# Utrain = st.zscore(Utrain, axis=2)
# Utrain = np.nan_to_num(Utrain)

fullorder_suffix = input_suffix #+ '_N3000'
FullOrder = np.load(sformat('FullOrder', fullorder_suffix, '.npz'))

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
Ufull = FullOrder['displacements_last_sim']
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

import pandas as pd
import seaborn as sns
from sklearn import preprocessing

x1 = dmaps.reduced_coordinates[1:,:].T
x2 = pca.reduced_coordinates.T

x1 = preprocessing.scale(x1)
x2 = preprocessing.scale(x2)

df1 = pd.DataFrame(x1)
df2 = pd.DataFrame(x2)

df1.add_prefix("X")
df2.add_prefix("X")
g = sns.PairGrid(df1)
g.map_diag(sns.kdeplot)
g.map_upper(sns.scatterplot, size=5)
g.map_lower(sns.kdeplot)
g.add_legend(title="", adjust_subtitles=True)

g = sns.PairGrid(df2)
g.map_diag(sns.kdeplot)
g.map_upper(sns.scatterplot, size=5)
g.map_lower(sns.kdeplot)
g.add_legend(title="", adjust_subtitles=True)

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

(timeline, Fx[0, -1])
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
reduced_step = 10
total_steps = timeline.shape[0]
ROM_displacements = np.empty((Nsim, model.total_DOFs, total_steps//reduced_step))
# =============================================================================
# ANALYSES
# =============================================================================

# =============================================================================
# PCA
# =============================================================================
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
    Ured = map_used.direct_transform_vector(newmark.displacements)
    ROM_displacements[case, :,:] = Ured[:, 1::reduced_step]


# =============================================================================
# DMAPS
# =============================================================================

dynamic.global_matrix_provider = providers.ReducedGlobalMatrixProvider(diff_map)
dmaps_displacements = np.empty((Nsim, model.total_DOFs, total_steps//reduced_step))


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
    Udmaps = diff_map.direct_transform_vector(newmark.displacements)
    dmaps_displacements[case, :,:] = Udmaps[:, 1::reduced_step]
end = time()
print("Finished in {:.3f} min".format(end/60 - start/60) )
# =============================================================================
# SAMPLING
# =============================================================================
if use_pca:
    method_suffix = 'pca/'
else:
    method_suffix = 'dmaps/'
output_suffix = input_suffix + '_ROM_' + method_suffix[:-2]

if Ntest>100:
    np.save(sformat('Displacements', output_suffix, '.npy'), ROM_displacements)
    
    
    
    save_dict = {
                'Ntrain' : Ntrain,
                'reduced_step' : reduced_step,
                'numeigs' : numeigs,
                'use_pca' : use_pca,
                'epsilon' : epsilon,
                'alpha' : alpha,
                'diff_time' : diff_time,
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
    
    
    np.savez(sformat('Metadata', output_suffix, '.npy'), **save_dict)

# pca r2 = 0.0431346999487219
# =============================================================================
# PLOTS
# =============================================================================

fig0, axes0 = plt.subplots(figsize=(8,6))

dof = 100
axes0.plot(timeline, Ufull[dof, :], label='FOM')

axes0.plot(timeline, Ured[dof, :], 
              label='PCA(num.eigs={:d})'.format(numeigs),
              linestyle='-.', color='g')

axes0.plot(timeline, Udmaps[dof, :],
              label='DMAPS(num.eigs={:d}, ε={:.1f}, diff.time={:d})'.format(numeigs, epsilon, diff_time),
              linestyle='--')

axes0.set_ylabel('$u$(mm)')
axes0.legend()
axes0.set_xlabel('$t$(sec)')
axes0.grid('on')
fig0.suptitle('Ntrain = {:d}, Sample step = {:d}, dof={:d}'.format(Ntrain, reduced_step, dof))
path = r"c:/users/constatza/Documents/thesis/LatexTemp/Figures/examples/Dynamic/"
filename = 'compare_timeline_dof{:d}_k{:d}.png'.format(dof, numeigs)
plt.savefig(os.path.join(path, filename), format='png', dpi=300, papertype='a4')
path = r"c:/users/constatza/Documents/thesis/LatexTemp/Figures/examples/Dynamic/"

path += method_suffix  


# smartplot.paper_style()

# fig = plt.figure(figsize=(5,4))

# ax = fig.add_subplot(111)
# e = np.logspace(-1, 3, num=50)
# Me = dmaps.kernel_sums_per_epsilon(epsilon=e)
# plt.loglog(e, Me)
# ax.axvline(epsilon, color='r', linestyle='--')
# ax.set_ylabel('$M(\epsilon)$')
# ax.set_xlabel('$\epsilon$')
# filename = 'Me.png'
# plt.savefig(os.path.join(path, filename), format='png', dpi=300, papertype='a4')  

# fig1, ax1 = plt.subplots(figsize=(5,4))
# # smartplot.plot_eigenvalues(dmaps.eigenvalues,
# #                             ax=ax1,
# #                             log=False,
# #                             marker='.',
# #                             label='DMAPS')
# ax1.set_ylabel('$\lambda_i$')
# ax1.set_xlabel('$i$')
# smartplot.plot_eigenvalues(pca.eigenvalues,
#                             ax=ax1,
#                             log= True,
#                             marker='.', 
#                             label='PCA')
# from matplotlib.ticker import MaxNLocator
# ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax1.legend()
# filename = 'eigs_N80_SS_200.png'
# plt.savefig(os.path.join(path, filename), format='png', dpi=300, papertype='a4')  



howmany = 4 
howmany = howmany * (numeigs > howmany) + (numeigs<= howmany) * numeigs
length = 1000

# bw = 0.5
# size = 4
# dmaps_frame = pd.DataFrame(dmaps.znormed_eigenvectors[:howmany, :].T)
# g = sns.PairGrid(dmaps_frame)
# g.map_upper(plt.scatter, marker='.', s=size)
# g.map_diag(sns.kdeplot, bw=5*bw)
# g.map_lower(sns.kdeplot, bw=5*bw)
# fig2 = g.fig
# fig2.suptitle('DMAPS Normalized Eigenvectors')


# pca_frame = pd.DataFrame(pca.znormed_eigenvectors[:howmany, :].T)
# g = sns.PairGrid(pca_frame)
# g.map_upper(plt.scatter, marker='.', s=size)
# g.map_diag(sns.kdeplot, bw=bw)
# g.map_lower(sns.kdeplot, bw=1.5*bw)
# fig3 = g.fig
# fig3.suptitle('PCA Normalized Eigenvectors')

# psi = [0,1,2]
# # set up a figure twice as wide as it is tall
# fig4 = plt.figure(figsize=plt.figaspect(3))
# ax41 = fig4.add_subplot(3, 1, 1, projection='3d')
# ax42 = fig4.add_subplot(3, 1, 2, projection='3d')
# ax43 = fig4.add_subplot(3, 1, 3, projection='3d')
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


