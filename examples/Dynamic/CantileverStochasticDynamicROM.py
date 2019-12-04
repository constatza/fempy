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
plt.close('all')
epsilon = 3
alpha = 1
numeigs = 10
displacements = np.random.randn(200, 100, 200)

# =============================================================================
# SAMPLING
# =============================================================================
u_train = displacements[::4, :, :].reshape(200,-1)

# =============================================================================
# DMAPS
# =============================================================================
dmaps = ml.DiffusionMap(u_train, epsilon=epsilon, alpha=alpha)
dmaps.fit(numeigs=numeigs, t=1) 
linear_dmaps = ml.LinearMap(domain=dmaps.reduced_coordinates, codomain=u_train)
u_dm = linear_dmaps.direct_transform_vector(dmaps.reduced_coordinates) 

# =============================================================================
# PCA
# =============================================================================
pca = ml.PCA(u_train)
pca.fit(numeigs=numeigs)
pca_map = ml.LinearMap(domain= pca.reduced_coordinates, codomain=u_train)
u_pca = pca_map.direct_transform_vector(pca.reduced_coordinates)

# =============================================================================
# PLOTS
# =============================================================================
#fig = plt.figure()
#ax = fig.add_subplot(111)
#e = np.logspace(-0, 2, num=20)
#plt.loglog(e, dmaps.kernel_sums_per_epsilon(u_train, e))