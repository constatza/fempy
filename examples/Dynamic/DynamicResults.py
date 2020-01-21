# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:33:21 2019

@author: constatza
"""
import gc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import mathematics.stochastic as st
import smartplot as sp
sp.paper_style()

times = [49, 99]
dofs = [2499]

times = np.array(times, dtype=int)
dofs = np.array(dofs, dtype=int) 

Times, Dofs = np.meshgrid(times, dofs)
Uf_data = np.load("Displacements_InertiaLoadXY.npy", mmap_mode='r')
Ur_data_pca = np.load("Displacements_InertiaLoadXY_ROM_PCA.npy", mmap_mode='r')
Ur_data_dmap = np.load("Displacements_InertiaLoadXY_ROM_dmap.npy", mmap_mode='r')
Uf = Uf_data[:, Dofs, Times]
Upca = Ur_data_pca[:, Dofs, Times]
Udmap = Ur_data_dmap[:, Dofs, Times]



b = .2
clip = (-b, b)
bw = b/40
fig, axes = plt.subplots(Dofs.shape[1], Dofs.shape[0], figsize=(5, 7))
fig.suptitle('Ntrain = {:d}, Sample step = {:d}, Neigs = {:d}'.format(10,10*10,5))
axes = axes.ravel()

for i,dof in enumerate(dofs):
    for j,time in enumerate(times):
         
        ax = axes[i+j*(i+1)]
        title = 't = {:.1f} sec, dof = {:d}'.format((time+1)*10/500, dof)
        # sns.kdeplot(Uf[:, i, j]-Upca[:, i, j], ax=ax, bw=bw, label='FOM', clip=clip)
        sns.kdeplot(Uf[:, i, j]-Upca[:, i, j], ax=ax, bw=bw, label='errors - PCA', clip=clip,
                    linestyle='-.')
        sns.kdeplot(Uf[:, i, j]-Udmap[:, i, j], ax=ax, bw=bw, label='errors - DMAPs', clip=clip,
                    linestyle='--')
        ax.set_title(title)

plt.legend()

import os 
path = "C:/Users/constatza/Documents/thesis/LatexTemp/Figures/examples/"
filename = 'pdf_errors_dof{:d}.png'.format(dof)
plt.savefig(os.path.join(path, filename), format='png', dpi=200, papertype='a4')        


