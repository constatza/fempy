# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:24:36 2019

@author: constatza
"""


import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def assign_element_displacements(displacements, model):
    # structure: dict<node_ID, dict<DOFtype, number>>
    nodal_displacements_dictionary = deepcopy(model.nodal_DOFs_dictionary)
    
    counter = -1
    for node in model.nodes:
        for DOFtype in model.nodal_DOFs_dictionary[node.ID].keys():
            if model.nodal_DOFs_dictionary[node.ID][DOFtype] != -1:
                counter += 1
                nodal_displacements_dictionary[node.ID][DOFtype] = displacements[counter]
            else: 
                nodal_displacements_dictionary[node.ID][DOFtype] = 0
    model.nodal_displacements_dictionary = nodal_displacements_dictionary    
    
    
    for element in model.elements:
        numnodes = len(element.DOFtypes)
        num_nodal_DOFtypes = len(element.nodal_DOFtypes)
        displacements = np.empty((numnodes,num_nodal_DOFtypes))
        for i, node in enumerate(element.nodes):
            for DOFtype in model.nodal_DOFs_dictionary[node.ID].keys():
                displacements[i, DOFtype-1] = model.nodal_displacements_dictionary[node.ID][DOFtype]
        element.node_displacements = displacements
    
    return model
                

def draw_deformed_shape(ax=None, elements=None, scale=100, *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    for element in elements:
        X = element.node_coordinates[:, 0]
        Y = element.node_coordinates[:, 1]
        X = np.append(X, X[0])
        Y = np.append(Y, Y[0])
        dX = element.node_displacements[:, 0]
        dY = element.node_displacements[:, 1]
        dX = np.append(dX, dX[0])
        dY = np.append(dY, dY[0])
        
        ax.plot(X + scale*dX, Y + scale*dY, *args, **kwargs)
    return ax

def save_forces(model):
    pass