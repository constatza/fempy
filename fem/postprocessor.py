# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:24:36 2019

@author: constatza
"""


import numpy as np
import matplotlib.pyplot as plt


def model_to_elements(model):
    for element in model.elements:
        element.nodes['displacementsX'] = model.displacements[element.nodes.dofX]
        element.nodes['displacementsY'] = model.displacements[element.nodes.dofY]
        element.compute_final_position()
    return model

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