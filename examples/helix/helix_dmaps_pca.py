# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""
import  numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fempy.dmaps import *
import fempy.smartplot as sp

plt.close('all')
sp.formal_serif()

epsilon = 10

timesteps = 1
numeigs = 2

sigma = 0.05
t = np.linspace(0, 3.5*np.pi, 200)
x = np.cos(t)*(1 + np.random.normal(scale=sigma, size=len(t)) )
y = np.sin(t)*(1 + np.random.normal(scale=sigma, size=len(t)) )
z = t*(1 + np.random.normal(scale=sigma, size=len(t)) )
U = np.concatenate([[x],[y],[z]])

U, Umean, Ustd = normalize(U)


"""Diffusion Maps"""   

eigvals, eigvecs = diffusion_maps(U, epsilon=epsilon, t=timesteps, numeigs=numeigs+1)
k = len(eigvals[eigvals>0.05]) + 1
#k = numeigs + 1
Fi =  eigvals[1:] * eigvecs[:, 1:]

A,res = least_squares(U, Fi)
print(A.shape)
Unew = A @ Fi.T
Unew = denormalize(Unew, Umean, Ustd)
xnew = Unew[0,:]
ynew = Unew[1,:]
znew = Unew[2,:]

"""PCA"""

Sigma = (U.T @ U)

val, vec = pca(Sigma, numeigs=numeigs)

m = len(val[val>[.05]]) 
m = numeigs
Lr = vec[:, :m]
P, res2 = least_squares(U, Lr) 

U_new2 = P @ Lr.T
U_new2 = denormalize(U_new2, Umean, Ustd)
x_pca = U_new2[0, :]
y_pca = U_new2[1, :]
z_pca = U_new2[2, :]

"""Plots"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, alpha=0.5, label='Data')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

e = np.logspace(-3,3, num=20)
cumsum = M(U, epsilon=e)
plt.figure()  
plt.loglog(e, cumsum)

plt.figure()
plt.plot(eigvals, '.-', label='Diffusion Maps')
plt.plot(val/np.max(val), '+-', label='PCA')
plt.ylabel('eigenvalues')
plt.legend()
plt.grid()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.scatter(Fi[:,0], Fi[:,1])
#ax2.scatter(Lr[:,0], Lr[:,1])
plt.grid()
plt.ylabel('$\Psi_2$')
plt.xlabel('$\Psi_1$')
plt.show() 

ax.scatter(xnew,ynew,znew, color='g', label='Diffusion Maps')  
ax.scatter(x_pca,y_pca,z_pca, color='r', label='PCA')
ax.legend()