# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:56:25 2019

@author: constatza
"""
import numpy as np
import matplotlib.pyplot as plt
from fem.core.entities import Model, Node
from fem.core.elements import Quad4


def rectangular_mesh_model(xlim, ylim, xnumel, ynumel, element_type):
    """ Builds a rectangular mesh model. 
            Nodes are numbered from left to right, bottom to top, starting from
            bottom left corner.
        
    Parameters
    ----------
    xlim : list
        List of length 2 containing the lower and upper limit of x-axis.
    ylim : list
        List of length 2 containing the lower and upper limit of y-axis.
    xnumel : int
        Number of elements along x-axis.
    ynumel : int
        Number of elements along y-axis.
    """
    numels = xnumel*ynumel
    xnumnodes = xnumel + 1
    ynumnodes = ynumel + 1
    
    # xlength = xlim[1] - xlim[0]
    # ylength = ylim[1] - ylim[0]
    
    # xstep = xlength/xnumel
    # ystep = ylength/ynumel
    
    xcoordinates = np.linspace(xlim[0], xlim[1], xnumnodes)
    ycoordinates = np.linspace(ylim[0], ylim[1], ynumnodes)
    
    nodeID = -1
    nodes_dictionary = {}
    for i in range(ynumnodes):
        for j in range(xnumnodes):
            nodeID +=1
            nodes_dictionary[nodeID] = Node(ID=nodeID,
                                                  X=xcoordinates[j],
                                                  Y=ycoordinates[i],
                                                  Z=0)           
    elements_dictionary = {}
    for i in range(numels):
                  
        elements_dictionary[i] = Quad4(ID=i,
                                       material=element_type.material,                              
                                       element_type=element_type, 
                                       thickness=element_type.thickness)
    elementID = -1
    for i in range(xnumel):
        for j in range(ynumel):
            elementID += 1
            lower_left_nodeID = xnumnodes*j + i 
            elements_dictionary[elementID].add_node(nodes_dictionary[lower_left_nodeID])       
            elements_dictionary[elementID].add_node(nodes_dictionary[lower_left_nodeID+1])
            elements_dictionary[elementID].add_node(nodes_dictionary[lower_left_nodeID+1+xnumnodes])
            elements_dictionary[elementID].add_node(nodes_dictionary[lower_left_nodeID+xnumnodes])
            

    model = Model(nodes_dictionary=nodes_dictionary, 
                  elements_dictionary=elements_dictionary)
    return model                      
    

# def read_nodes(file='nodes.csv'): 
#     return pd.read_csv(file)


# def read_connectivity(file='connectivity.csv'):
#     return pd.read_csv(file, header=None, usecols=(1,2,3,4))

def draw_mesh(elements, *args, **kwargs):
    
    for element in elements:
        X = [node.X for node in element.nodes]
        Y = [node.Y for node in element.nodes]
        X.append(X[0])
        Y.append(Y[0])
        plt.plot(X, Y, *args, **kwargs)
    plt.show()
            

