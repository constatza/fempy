# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:03:35 2019

@author: constatza
"""

import matplotlib.pyplot as plt
import numpy as np
import fempy.mathematics.manilearn as ml
import fempy.mathematics.statistics as stat
import fempy.smartplot as smartplot

from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from scipy.stats import zscore




plt.close('all')

epsilon = 1
alpha = 0
timesteps = 1
numeigs = 2

# set parameters
length_phi = 10 #length of swiss roll in angular direction
length_Z = 10#length of swiss roll in z direction
sigma = 2 #noise strength
m = 2000 #number of samples
# create dataset
phi = length_phi/6 *np.random.randn(m) + length_phi*.6 # normal distribution phi
#phi = length_phi *np.random.rand(m) #+ length_phi*.6

X = length_Z*np.random.randn(m)
Y = (phi )*np.sin(phi)
Z = (phi )*np.cos(phi)
swiss_roll = np.array([X, Y, Z])
U = np.concatenate([[X],[Y],[Z]])
U = zscore(U, axis=1)
d = U.shape[0]
N = U.shape[1]
Phi = np.tile(phi, (numeigs, 1))
color = phi#np.arange(swiss_roll.shape[1])
X_r, err = manifold.locally_linear_embedding(U.T, n_neighbors=20,
                                             n_components=numeigs)
Xr = X_r.T
#----------------------------------------------------------------------
# Plot result



"""Diffusion Maps"""   

dmaps = ml.DiffusionMap(U, epsilon=epsilon, alpha=alpha)
dmaps.fit(numeigs=numeigs, t=1) 

linear_dmaps = ml.LinearMap(domain=dmaps.reduced_coordinates, 
                                codomain=U)

res_dm = linear_dmaps.residuals
   
U_dm = linear_dmaps.direct_transform_vector(dmaps.reduced_coordinates) 
x_dm = U_dm[0,:]
y_dm = U_dm[1,:]
z_dm = U_dm[2,:]
   
"""PCA"""
pca = ml.PCA(U)
pca.fit(numeigs=numeigs)

pca_map = ml.LinearMap(domain= pca.reduced_coordinates,
                          codomain=U)

U_pca = pca_map.direct_transform_vector(pca.reduced_coordinates)
x_pca = U_pca[0, :]
y_pca = U_pca[1, :]
z_pca = U_pca[2, :]

"""Plots"""
x = zscore(X)
y = zscore(Y)
z = zscore(Z)

fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection='3d')
ax0.scatter(x, y, z, alpha=0.5, label='Original Data', c=phi, marker='.')

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(x, y, z, alpha=0.5, label='Original Data', color='b', marker='.')
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
plt.grid()


 
fig4, axes4 = plt.subplots(1, 3)
fig4.suptitle('Normalized Eigenvectors')
smartplot.plot_eigenvectors(dmaps.reduced_coordinates.T, ax=axes4[0], title='DMAPS', c=phi, marker='.')
smartplot.plot_eigenvectors(Xr.T, ax=axes4[1], title='LLE', c=color.T, marker='.')
smartplot.plot_eigenvectors(pca.reduced_coordinates.T, ax=axes4[2], title='PCA', c=color, marker='.')

 
fig5, axes5 = plt.subplots(1, 3)
fig5.suptitle('Correlation')

axes5[0].plot(phi, dmaps.reduced_coordinates.T, marker='.', linestyle='')
axes5[1].plot(phi, Xr.T, marker='.', linestyle='')
axes5[2].plot(phi, pca.reduced_coordinates.T, marker='.', linestyle='')

correl_dfm = np.corrcoef(x=phi, y=dmaps.reduced_coordinates)
correl_lle = np.corrcoef(x=phi, y=Xr)
correl_pca = np.corrcoef(x=phi, y=pca.reduced_coordinates)
