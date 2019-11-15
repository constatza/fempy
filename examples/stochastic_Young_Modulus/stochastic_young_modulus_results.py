# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:33:58 2019

@author: constatza
"""

import numpy as np 
import matplotlib.pyplot as plt
import fempy.smartplot as smartplot
import fempy.mathematics.manilearn as ml

from scipy.stats import zscore
import fempy.mathematics.statistics as stat


plt.close('all')

# =============================================================================
# Input
# =============================================================================

epsilon = 10
alpha = 1
numeigs = 5
timesteps = 3
Nsim = 50

dof = 1000
filename = r"stochastic_E_displacements_20x50.npy"
stiffname = r"stochastic_K.npy"

# =============================================================================
# Preparations
# =============================================================================

displacements = np.load(filename)
#Ks = np.load(stiffname)

displacements = np.transpose(displacements)
data = stat.StochasticField(data=displacements)

# subset of displacements
training_data = stat.StochasticField(data=displacements[:, :Nsim]) 
color = training_data.data[1,:]
# =============================================================================
# Diffusion Maps
# =============================================================================
dmap = ml.DiffusionMap(training_data.data,epsilon=epsilon, alpha=alpha)
dmap.fit(numeigs=numeigs, t=timesteps)

dmap_to_origin = ml.LinearMap(domain = dmap.reduced_coordinates,
                              codomain = training_data.data)

Unew_dm = dmap_to_origin.direct_transform_vector(dmap.reduced_coordinates)
#Unew_dm = stat.zscore_inverse(Unew_dm, training_data.mean, Ustd)
x_dm = Unew_dm[2*dof-2, :]
y_dm = Unew_dm[2*dof-1, :]

# =============================================================================
# PCA
# =============================================================================
pca = ml.PCA(training_data.data)
pca.fit(numeigs=numeigs)
pca_to_origin = ml.LinearMap(domain=pca.reduced_coordinates,
                             codomain=training_data.data)

Unew_pca = pca_to_origin.direct_transform_vector(pca.reduced_coordinates)
#Unew_pca = stat.zscore_inverse(Unew_pca, training_data.mean, training_data.std)
x_pca = Unew_pca[2*dof-2, :]
y_pca = Unew_pca[2*dof-1, :]




# =============================================================================
# Tests
# =============================================================================

stiffness_matrix = Ks[:, :, -1]
u_true = displacements[:,-1]
force_vector = np.zeros((stiffness_matrix.shape[0],1))
force_vector[-2,0] = 100

reduced_matrix_dm = dmap_to_origin.inverse_transform_matrix(stiffness_matrix)
reduced_vector_dm = dmap_to_origin.inverse_transform_vector(force_vector)

reduced_displacements_dm = np.linalg.solve(reduced_matrix_dm, reduced_vector_dm)

u_dm = dmap_to_origin.direct_transform_vector(reduced_displacements_dm)
u_dm = np.squeeze(u_dm)
errors_dm = np.squeeze(u_dm) - u_true
errnorm_dm = np.linalg.norm(errors_dm)
relative_err_dm = np.abs(errors_dm)



reduced_matrix_pca = pca_to_origin.inverse_transform_matrix(stiffness_matrix)
reduced_vector_pca = pca_to_origin.inverse_transform_vector(force_vector)

reduced_displacements_pca = np.linalg.solve(reduced_matrix_pca, reduced_vector_pca)

u_pca = pca_to_origin.direct_transform_vector(reduced_displacements_pca)
u_pca = np.squeeze(u_pca)

errors_pca = np.squeeze(u_pca) - u_true
errnorm_pca = np.linalg.norm(errors_pca)
relative_err_pca = np.abs(errors_pca)

"""Plots"""
# plot M(e)

plt.close('all')
epsilons = np.logspace(-1.5, 4, 30)
plt.figure()  
plt.loglog(epsilons, dmap.kernel_sums_per_epsilon(epsilon=epsilons))


x, y = training_data.data[2*dof-2, :], training_data.data[2*dof-1, :]
fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.scatter(x, y, alpha=0.5, label='Data', marker='x')
ax.scatter(x_dm, y_dm, color='g', label='Diffusion Maps', marker='.')  
ax.scatter(x_pca, y_pca, color='r', label='PCA', marker='.')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')


fig3, ax3 = plt.subplots(1,2)
smartplot.plot_eigenvalues(dmap.eigenvalues, ax=ax3[0], color='b', marker='.')
smartplot.plot_eigenvalues(pca.eigenvalues, ax=ax3[1], color='g', marker='+')


fig4, ax4 = plt.subplots(1,2)
smartplot.plot_eigenvectors(dmap.eigenvectors[1:3,:], ax=ax4[0], title='Diffusion Maps', c=color)
smartplot.plot_eigenvectors(pca.eigenvectors[0:2, :], ax=ax4[1], title='PCA', c=color)

plt.figure()
smartplot.histogram(relative_err_pca, norm_hist=True)
plt.draw()