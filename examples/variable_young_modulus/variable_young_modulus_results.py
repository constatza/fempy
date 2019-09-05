# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:33:58 2019

@author: constatza
"""

import numpy as np 
import matplotlib.pyplot as plt
import fempy.dmaps as dm
from mpl_toolkits.mplot3d import Axes3D

filename = r"variable_E_displacements_20x50.npy"

with open(filename) as file:
    displacements = np.load(filename)

displacements = np.transpose(displacements)


epsilon = 5
numeigs = 10
timesteps = 1
Nsim = 10

eigvals, eigvecs = dm.diffusion_maps(displacements[:,:Nsim], epsilon=epsilon, t=timesteps, k=numeigs)
Fi = -eigvals* eigvecs


k = len(eigvals[eigvals>0.1])
A, res = dm.ls_approx(displacements[:, :Nsim], Fi[:,:k])

print(A.shape)

Unew = A @ Fi[:, :k].T
xnew = Unew[0,:]
ynew = Unew[1,:]
znew = Unew[2,:]

#epsilons = np.logspace(-3,3)
#M = dm.M(displacements, epsilon=epsilons)
#plt.figure()  
#plt.loglog(epsilon, M)

plt.figure()
plt.plot(eigvals, 'o-')
plt.ylabel('eigenvalues')
plt.grid()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(Fi[:,1], Fi[:,2], Fi[:,3])
plt.grid()
plt.ylabel('Ψ2')
plt.xlabel('Ψ1')
plt.show() 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(displacements[0, :Nsim],displacements[1, :Nsim], displacements[2, :Nsim])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(xnew,ynew,znew, color='g')     