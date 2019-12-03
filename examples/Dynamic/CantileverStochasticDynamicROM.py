# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:03:04 2019

@author: constatza
"""
import mathematics.manilearn as ml
import numpy as np
import matplotlib.pyplot as plt
#analyzer = depickle ...
#displacements = load...
#u_train = subset of dis...

epsilon = 1
u_train = np.random.randn(2100, 200)
dmaps = ml.DiffusionMap(u_train, epsilon=epsilon, alpha=0)

fig = plt.figure()
ax = fig.add_subplot(111)
e = np.logspace(-3, 3, num=20)
plt.loglog(e, dmaps.kernel_sums_per_epsilon(u_train, e))