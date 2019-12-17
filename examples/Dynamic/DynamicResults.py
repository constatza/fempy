# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:33:21 2019

@author: constatza
"""
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mathematics.stochastic as st


times = [15, 82]
dofs = [2099, 2100]

times = np.array(times, dtype=int)
dofs = np.array(dofs, dtype=int) -1

Times, Dofs = np.meshgrid(times, dofs)
U = np.load("U1_N2500.npy", mmap_mode='r')

Ur = U[Dofs, Times, :]


plt.figure()
for i,dof in enumerate(dofs):
    for j,time in enumerate(times):
        label = 't={:.1f}, dof={:d}'.format(time, dof+1)
        field = st.StochasticField(data=Ur[i, j, :])
        data = field.data
        sns.kdeplot(data, bw=0.05, label=label,  clip=(-.4, 0.4))
plt.legend()