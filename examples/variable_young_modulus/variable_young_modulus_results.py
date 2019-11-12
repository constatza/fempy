# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:33:58 2019

@author: constatza
"""

import numpy as np 
import matplotlib.pyplot as plt
import fempy.dmaps as dm
from mpl_toolkits.mplot3d import Axes3D
from fempy import smartplot

"""
Input
"""
epsilon = 40
numeigs = 3
timesteps = 10
Nsim = 2000
dof = 1050

plt.close('all')
filename = r"variable_E_displacements_20x50.npy"
with open(filename) as file:
    displacements = np.load(filename)

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
A_dm, res_dm = dm.least_squares(U, Fi)

print(A_dm.shape)

Unew_dm = A_dm @ Fi[:, :k].T
Unew_dm = dm.denormalize(Unew_dm, Umean, Ustd)
x_dm = Unew_dm[2*dof-2, :]
y_dm = Unew_dm[2*dof-1, :]




"""PCA"""

eigvals_pca, eigvecs_pca = dm.pca(U, numeigs=numeigs)

m = len(eigvals_pca[eigvals_pca>[.05]]) 
m = numeigs
Lr = eigvecs_pca[:, :m]
A_pca, res_pca = dm.least_squares(U, Lr) 

Unew_pca = A_pca @ Lr.T 
Unew_pca = dm.denormalize(Unew_pca, Umean, Ustd)
x_pca = Unew_pca[2*dof-2, :]
y_pca = Unew_pca[2*dof-1, :]



plt.figure()
plt.plot(eigvals_dm, 'o-', label='Diffusion Maps')
plt.plot(eigvals_pca/np.max(eigvals_pca), 'x-', label='PCA')
plt.ylabel('Eigenvalues')
plt.legend()
plt.grid()

colorange = range(Fi.shape[0])
ax3 = smartplot.eigenvector_plot(Fi[:,:3], title='Eigenvectors - Diffusion Maps', c=colorange)
plt.colorbar()
ax4 = smartplot.eigenvector_plot(Lr[:, :3], title='Eigenvectors - PCA', c=colorange)

plt.colorbar()

   