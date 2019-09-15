# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:33:58 2019

@author: constatza
"""

import numpy as np 
import matplotlib.pyplot as plt
import fempy.dmaps as dm
from mpl_toolkits.mplot3d import Axes3D

"""
Input
"""
epsilon = 10
numeigs = 12
timesteps = 1
Nsim = 200

filename = r"stochastic_E_displacements_20x50.npy"
with open(filename) as file:
    displacements = np.load(filename)

displacements = np.transpose(displacements)
U = displacements[:, :Nsim]
utop = displacements[:,:]
utop_norm1 = np.linalg.norm(utop, 1, axis=0)

eigvals, eigvecs = dm.diffusion_maps(displacements[:,:Nsim],
                                     epsilon=epsilon, 
                                     t=timesteps, 
                                     numeigs=numeigs)
Fi = eigvals* eigvecs

k = len(eigvals[eigvals>0.05])
A, res = dm.ls_approx(displacements[:, :Nsim], Fi[:,:k])

print(A.shape)

Unew = A @ Fi[:, :k].T

errors = Unew - U
errnorm = np.linalg.norm(errors)
relative_err = (errors/U)

#epsilons = np.logspace(-3,3, 10)
#M = dm.M(displacements, epsilon=epsilons)
#plt.figure()  
#plt.loglog(epsilon, M)

plt.figure()
plt.plot(eigvals, 'o-')
plt.ylabel('eigenvalues')
plt.grid()



   