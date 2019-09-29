# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:16:12 2019

@author: constatza
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""
import  numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fempy.dmaps import *

plt.close('all')

epsilon = 1

timesteps = 1
numeigs = 3

sigma = 0.05
r1 = 4
r2 = 0.5
t1 = np.linspace(0, 2, 40)
t2 = np.linspace(0, 2, 20)
t1, t2 = np.meshgrid(t1,t2)
#x = np.cos(t1)*(r1 + r2*np.cos(t2))
#y = np.sin(t1)*(r1 + r2*np.cos(t2))
z = np.sin(t1 + t2) *np.exp(-(t2-1)**2-(t1-1)**2)
x = t1.ravel()
y = t2.ravel()
z = z.ravel()
U = np.concatenate([[x],
                    [y],
                    [z]])
U, Umean, Ustd = normalize(U)


"""Diffusion Maps"""   


eigvals, eigvecs = diffusion_maps(U, epsilon=epsilon, t=timesteps, numeigs=numeigs+1)
k = len(eigvals[eigvals>0.05]) 
k = numeigs + 1
Fi =  eigvals * eigvecs

A,res = ls_approx(U, Fi)
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
P, res2 = ls_approx(U, Lr) 

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
plt.plot(eigvals/np.max(eigvals), '+-', label='Diffusion Maps')
plt.plot(val/np.max(val), 'x-', label='PCA')
plt.ylabel('eigenvalues')
plt.grid()
plt.legend()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(Fi[:,1], Fi[:,2], Fi[:,3], label='DM')
#ax2.scatter(Lr[:,0], Lr[:,1], Lr[:,2], label='PCA')
ax2.legend()
plt.grid()
plt.ylabel('$\Psi_2$')
plt.xlabel('$\Psi_1$')
plt.show() 

ax.scatter(xnew,ynew,znew, color='g', label='Diffusion Maps')  
ax.scatter(x_pca,y_pca,z_pca, color='r', label='PCA')
ax.legend()