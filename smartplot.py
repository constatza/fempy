# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:56:44 2019

@author: constatza
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def histogram(x, *args, **kwargs):
    sns.set()
    return sns.distplot(x, *args, **kwargs)

def formal_serif():    
    plt.rc( 'text', **{'usetex' : True})
    # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})

    # Set the font used for MathJax - more on this later
    plt.rc('mathtext',**{'default':'regular'})
    
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)


def eigenvector_plot(F, title=None,label=None):
    numeigs = F.shape[1]
    fig = plt.figure()
    if numeigs==2:
        
        ax = fig.add_subplot(111)
        lines = ax.scatter(F[:, 0], F[:, 1])
        ax.set_xlabel('$\psi_1$')
        ax.set_ylabel('$\psi_2$')

    elif numeigs==3:
        ax = fig.add_subplot(111, projection='3d')
        lines = ax.scatter(F[:,0], F[:, 1], F[:, 2])
        ax.set_xlabel('$\psi_1$')
        ax.set_ylabel('$\psi_2$')
        ax.set_zlabel('$\psi_3$')
        
    else:
        ax = fig.add_subplot(111)
        lines = ax.plot(F)
        ax.legend(lines, list(np.arange(F.shape[1])+1))

    ax.set_title(title)
    ax.grid()
    return ax

if __name__ == "__main__":
    plt.close('all')
    formal_serif()
    F = np.reshape(np.arange(20), (5,-1) )   
    
    ax = eigenplot(F, title='Nice')
    
    plt.show()

