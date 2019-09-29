# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:56:44 2019

@author: constatza
"""
import matplotlib as plt


def formal_serif():    
    plt.rc( 'text', **{'usetex' : True})
    # Set the global font to be DejaVu Sans, size 10 (or any other sans-serif font of your choice!)
    plt.rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})

    # Set the font used for MathJax - more on this later
    plt.rc('mathtext',**{'default':'regular'})
    
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)