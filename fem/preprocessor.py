# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:56:25 2019

@author: constatza
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')


def read_nodes(file='nodes.csv'):
    nodes = pd.read_csv(file)
    return create_node_dofs(nodes)


def read_connectivity(file='connectivity.csv'):
    return pd.read_csv(file, header=None, usecols=(1,2,3,4))


def create_node_dofs(nodes):
    try:
        nodes['Xrestrained'].fillna(False, inplace=True)
        nodes['Yrestrained'].fillna(False, inplace=True)
        nodes['Fx'].fillna(0, inplace=True)
        nodes['Fy'].fillna(0, inplace=True)
        nodes['Dx'].fillna(0, inplace=True)
        nodes['Dy'].fillna(0, inplace=True)
    except:
        print('exception!!!!@#$')    
    nodes['dofX'] = 2*nodes.index.values
    nodes['dofY'] = 2*nodes.index.values + 1
    nodes.set_index('Node', inplace=True)
    return nodes


def draw_underformed_shape(ax=None, elements=None, what='', *args, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    for element in elements:
        x = np.concatenate([element.nodes[what + 'X'],
                            element.nodes.X.head(1)])
        y = np.concatenate([element.nodes[what + 'Y'],
                            element.nodes.Y.head(1)])
        ax.plot(x, y, *args, **kwargs)
    return ax
