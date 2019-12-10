# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import mathematics.manilearn as ml
# plt.close('all')
np.random.seed(10)
t = np.random.randn(2100)*np.pi/3 + np.pi/2


x = np.cos(t)
y = np.sin(t)    


dmap = ml.DiffusionMap(np.array([x,y]), epsilon=.5, alpha=0)

dmap.fit(numeigs=2, t=10)

U = dmap.reduced_coordinates
u, v = U[1,:], U[2,:]


fig, ax = plt.subplots()
mag = np.max([u,v])
plt.scatter(u/mag,v/mag, c=t)
ax.set_aspect('equal', 'box')
plt.show()