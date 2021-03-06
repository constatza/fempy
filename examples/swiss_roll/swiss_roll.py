# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:03:35 2019

@author: constatza
"""

import matplotlib.pyplot as plt
import numpy as np
import mathematics.manilearn as ml
import mathematics.stochastic as stat
import smartplot as smartplot

from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from scipy.stats import zscore



np.random.seed(10)
plt.close('all')

epsilon = 1
alpha = 0
timesteps = 1
numeigs = 2

# set parameters
std_phi = 2 #length of swiss roll in angular direction
std_roll = 5#length of swiss roll in z direction
sigma = 2 #noise strength
m = 1000 #number of samples
# create dataset
# phi = np.linspace(1,9,m) # normal distribution phi
phi = np.random.normal(5, std_phi, m ) 

X = np.random.normal(0, std_roll, m ) 
Y = (phi )*np.sin(phi)
Z = (phi )*np.cos(phi) 
swiss_roll = np.array([X, Y, Z])
U = np.concatenate([[X],[Y],[Z]])
U = zscore(U, axis=1)
d = U.shape[0]
N = U.shape[1]
Phi = np.tile(phi, (numeigs, 1))
color = phi#np.arange(swiss_roll.shape[1])
# X_r, err = manifold.locally_linear_embedding(U.T, n_neighbors=20,
#                                             n_components=numeigs)
# Xr = X_r.T

ax = plt.subplot()
ax.scatter(phi,X, c=phi, s=10)
plt.grid()
plt.show()
ax.set_xlabel("$\phi$")
ax.set_ylabel("$t$")
# =============================================================================
# Diffusion Maps 
# =============================================================================

dmaps = ml.DiffusionMap(U, epsilon=epsilon, alpha=alpha)
dmaps.fit(numeigs=numeigs, t=1) 

linear_dmaps = ml.LinearMap(domain=dmaps.reduced_coordinates, 
                                codomain=U)

res_dm = linear_dmaps.res
   
U_dm = linear_dmaps.direct_transform_vector(dmaps.reduced_coordinates) 
x_dm = U_dm[0,:]
y_dm = U_dm[1,:]
z_dm = U_dm[2,:]
   
# =============================================================================
# PCA
# =============================================================================
pca = ml.PCA(U)
pca.fit(numeigs=numeigs)

pca_map = ml.LinearMap(domain= pca.reduced_coordinates, codomain=U)
#pca_map.matrix = pca.eigenvectors

U_pca = pca_map.direct_transform_vector(pca.reduced_coordinates)
x_pca = U_pca[0, :]
y_pca = U_pca[1, :]
z_pca = U_pca[2, :]

# =============================================================================
# Plots
# =============================================================================
X = zscore(X)
Y= zscore(Y)
Z = zscore(Z)

fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection='3d')
ax0.scatter(X, Y, Z, alpha=0.9, label='Original Data', c=phi, marker='.')
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.set_zlabel('Z')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(X, Y, Z, alpha=0.5, label='Original Data', color='b', marker='.')
ax1.scatter(x_dm,y_dm,z_dm, color='r', label='Diffusion Maps', marker='.')  
ax1.scatter(x_pca,y_pca,z_pca, color='g', label='PCA',  marker='.')
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
e = np.logspace(-3, 3, num=20)
plt.loglog(e, dmaps.kernel_sums_per_epsilon(U, e))

ax3 = smartplot.plot_eigenvalues(dmaps.eigenvalues, marker='+', label='DMAPS')
smartplot.plot_eigenvalues(pca.eigenvalues/np.max(pca.eigenvalues),ax=ax3, marker='o', label='PCA')
ax3.legend()
ax3.set_ylabel('$\lambda_i$')
plt.grid()


 
fig4, axes4 = plt.subplots(1, 2, sharey=True)
fig4.suptitle('Reduced Coordinates')
smartplot.plot_eigenvectors(dmaps.reduced_coordinates[1:,:], ax=axes4[0], title='DMAPS', c=color, marker='.')
#smartplot.plot_eigenvectors(Xr, ax=axes4[2], title='LLE', c=color, marker='.')
smartplot.plot_eigenvectors(pca.reduced_coordinates, ax=axes4[1], title='PCA', c=color, marker='.')

 
print(pca.eigenvalues/np.sum(pca.eigenvalues))
print(dmaps.eigenvalues/np.sum(dmaps.eigenvalues))