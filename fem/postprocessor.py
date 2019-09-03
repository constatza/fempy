# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:24:36 2019

@author: constatza
"""


import numpy as np
import matplotlib.pyplot as plt


def assign_element_displacements(displacements, model):
    counter = -1
    for element in model.elements:
        element_disp = []
        for i, node in enumerate(element.nodes):
            node_disp = []
            for DOF in model.nodal_DOFs_dictionary[node.ID].values():
            
                if DOF != -1:
                    counter += 1
                    node_disp.append(displacements[counter])
                else:
                    node_disp.append(0)
            element_disp.append[node_disp]
        element.displacements = element_disp
                    
                

def draw_deformed_shape(ax=None, elements=None, scale=100, *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    for element in elements:
        finalX = element.nodes.X + scale * element.nodes.displacementsX
        finalY = element.nodes.Y + scale * element.nodes.displacementsY
        x = np.concatenate([finalX, finalX.head(1)])
        y = np.concatenate([finalY, finalY.head(1)])
        ax.plot(x, y, *args, **kwargs)
    return ax

def save_forces(model):
    model.nodes['total_forcesX'] = model.total_external_forces[0::2]
    model.nodes['total_forcesY'] = model.total_external_forces[1::2]
    model.nodes['reactionsX'] = model.reactions[0::2]
    model.nodes['reactionsY'] = model.reactions[1::2]
    return model