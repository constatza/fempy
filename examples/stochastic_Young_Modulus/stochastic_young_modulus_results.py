# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:33:58 2019

@author: constatza
"""

import numpy as np 
import matplotlib.pyplot as plt
import fempy.dmaps as dm
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg
import fempy.smartplot as smartplot


plt.close('all')
"""
Input
"""
epsilon = 100
numeigs = 5
timesteps = 1
Nsim = 200
Ntests = Nsim

filename = r"stochastic_E_displacements_20x50.npy"
stiffname = r"stochastic_K.npy"

"""Preparations"""

displacements = np.load(filename)
#Ks = np.load(stiffname)
    
displacements = np.transpose(displacements)
U, Umean, Ustd = dm.normalize(displacements[:, :Nsim])
utop = displacements[:,:]
utop_norm1 = np.linalg.norm(utop, 1, axis=0)



"""Diffusion Maps"""
eigvals_dm, eigvecs_dm = dm.diffusion_maps(U,
                                     epsilon=epsilon, 
                                     t=timesteps, 
                                     numeigs=numeigs+1)


k = len(eigvals_dm[eigvals_dm>0.05])
k = numeigs+1
Fi =  eigvals_dm[1:] * eigvecs_dm[:, 1:]
A_dm, res_dm = dm.ls_approx(U, Fi)

print(A_dm.shape)

Unew_dm = A_dm @ Fi[:, :k].T
Unew_dm = dm.denormalize(Unew_dm, Umean, Ustd)
x_dm = Unew_dm[0, :]
y_dm = Unew_dm[1, :]
z_dm = Unew_dm[2, :]



"""PCA"""

eigvals_pca, eigvecs_pca = dm.pca(U, numeigs=numeigs)

m = len(eigvals_pca[eigvals_pca>[.05]]) 
m = numeigs
Lr = eigvecs_pca[:, :m]
A_pca, res_pca = dm.ls_approx(U, Lr) 

Unew_pca = A_pca @ Lr.T 
Unew_pca = dm.denormalize(Unew_pca, Umean, Ustd)
x_pca = Unew_pca[0, :]
y_pca = Unew_pca[1, :]
z_pca = Unew_pca[2, :]



"""Tests"""

stiffness_matrix = Ks[:, :, -1]
u_true = displacements[:,-1]
force_vector = np.zeros((stiffness_matrix.shape[0],1))
force_vector[-2,0] = 100

reduced_matrix_dm = A_dm.T @ stiffness_matrix @ A_dm
reduced_vector_dm = A_dm.T @ force_vector

reduced_displacements_dm = np.linalg.solve(reduced_matrix_dm, reduced_vector_dm)

u_dm = dm.denormalize(A_dm @ reduced_displacements_dm, Umean, Ustd)
u_dm = np.squeeze(u_dm)
errors_dm = np.squeeze(u_dm) - u_true
errnorm_dm = np.linalg.norm(errors_dm)
relative_err_dm = np.abs(errors_dm)



reduced_matrix_pca = A_pca.T @ stiffness_matrix @ A_pca
reduced_vector_pca = A_pca.T @ force_vector

reduced_displacements_pca = np.linalg.solve(reduced_matrix_pca, reduced_vector_pca)

u_pca = dm.denormalize(A_pca @ reduced_displacements_pca, Umean, Ustd)
u_pca = np.squeeze(u_pca)

errors_pca = np.squeeze(u_pca) - u_true
errnorm_pca = np.linalg.norm(errors_pca)
relative_err_pca = np.abs(errors_pca)

"""Plots"""
# plot M(e)
epsilons = np.logspace(-1.5, 4, 30)
M = dm.M(U, epsilon=epsilons)
plt.figure()  
plt.loglog(epsilons, M)

U = dm.denormalize(U, Umean, Ustd)
x, y, z = U[0, :], U[1, :], U[2, :]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, alpha=0.5, label='Data')
ax.scatter(x_dm, y_dm, z_dm, color='g', label='Diffusion Maps')  
ax.scatter(x_pca, y_pca, z_pca, color='r', label='PCA')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


plt.figure()
plt.plot(eigvals_dm/np.max(eigvals_dm), 'o-', label='Diffusion Maps')
plt.plot(eigvals_pca/np.max(eigvals_pca), 'x-', label='PCA')
plt.ylabel('Eigenvalues')
plt.legend()
plt.grid()


ax3 = smartplot.eigenvector_plot(Fi, title='Eigenvectors - Diffusion Maps')
ax4 = smartplot.eigenvector_plot(Lr, title='Eigenvectors - PCA')

plt.figure()

smartplot.histogram(relative_err_pca, norm_hist=True)
plt.draw()