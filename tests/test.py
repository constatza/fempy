# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import mathematics.manilearn as ml
import scipy as sp
plt.close('all')

t = np.random.randn(2100)*np.pi/3 + np.pi/2


r = sp.random.randn(2, 500)
    
theta = np.pi/4

Rot = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))

Scale = np.array((( 3, 0),
                 (0, 1)))

xy = Rot @ Scale @ r

pca = ml.PCA(xy)

pca.fit(numeigs=2)

U = pca.reduced_coordinates
u, v = U[0,:], U[1,:]


fig, ax = plt.subplots()
plt.scatter(xy[0,:], xy[1,:])
ax.set_aspect('equal', 'box')
plt.show()
np.savetxt('pca_scatter.csv', xy.T, delimiter=',')