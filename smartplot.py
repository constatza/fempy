# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:56:44 2019

@author: constatza
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import zscore 


def paper_style():
    plt.style.use(['seaborn-whitegrid', 'seaborn-paper'])
    font = {'family' : 'Times New Roman',
            'weight' : 'bold',}
            # 'size'   : 10}
    matplotlib.rc('image', cmap='viridis')
    matplotlib.rc('font', **font)


def gridplot(axes, iterable, titles=None, only2d=True, *args, **kwargs):
    """
    e.g. 
    fig, axes = subplots(2,2)
    gridplot(axes, (data1, data2, data3, data4))
    """
    for i, data in enumerate(iterable):
        
        axes[i] = plotnd(data, ax=axes[i],  *args, **kwargs)
        if titles is not None:
            axes[i].set_title[titles[i]]
    return axes
        
def plotnd(data, ax=None, *args, **kwargs):
    returnax = True
    if ax is None: 
        ax = plt
        returnax = False
    numdimensions = len(data)
    
    if numdimensions>1:
        if isinstance(data, list) or isinstance(data, tuple):
            data=np.array(data)
        
        ax.plot(data[0, :], data[1:,:].T, *args, **kwargs)
    elif numdimensions==1:
        ax.plot(data, *args, **kwargs)
    
    if returnax: return ax
    
         
         


def plot23d(x, y, z=None, ax=None, title=None, *args, **kwargs):
    if ax is None: ax = plt
    if z is None:
        ax.plot(x, y, *args, **kwargs)
    elif ax.projection=='3d':
        ax.plot(x, y, z, *args, **kwargs)
    else:
        print("Axis projection!='3d'")
    ax.set_title(title)
    return ax
                

def formal_serif():    
    plt.rc( 'text', **{'usetex' : True})
    # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})

    # Set the font used for MathJax - more on this later
    plt.rc('mathtext',**{'default':'regular'})
    
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)


def plot_eigenvalues(V, ax=None, log=False, *args, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    if not log:
        ax.plot(V, *args, **kwargs)
    else:
        ax.semilogy(V, *args, **kwargs)
    # ax.set_title('Eigenvalues')
    return ax



def plot_eigenvectors(F, ax=None, c=None, psi=(1,2), title=None, *args, **kwargs):
    
    F = zscore(F.T, axis=0)
    try:
        numeigs = F.shape[1]
    except IndexError:
        numeigs = 1
    
    
    if numeigs==2:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        lines = ax.scatter(F[:, 0], F[:, 1], c=c, *args, **kwargs)
        ax.set_xlabel('$\psi_{:d}$'.format(psi[0]))
        ax.set_ylabel('$\psi_{:d}$'.format(psi[1]))

    elif numeigs==3:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        lines = ax.scatter(F[:,0], F[:, 1], F[:, 2], c=c, *args, **kwargs)
        ax.set_xlabel('$\psi_{:d}$'.format(psi[0]))
        ax.set_ylabel('$\psi_{:d}$'.format(psi[1]))
        ax.set_zlabel('$\psi_{:d}$'.format(psi[2]))
        
    else:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        lines = ax.plot(F, linestyle='-', *args, **kwargs)
        ax.legend(lines, list(np.arange(numeigs)+1))
    if title is not None:
        ax.set_title(title)

    ax.grid()
    return ax


if __name__ == "__main__":
    plt.close('all')
    formal_serif()

