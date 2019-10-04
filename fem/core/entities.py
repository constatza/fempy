# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:44:21 2019

@author: constatza
"""
import numpy as np
from enum import IntEnum
from .assemblers import GenericDOFEnumerator

class DOFtype(IntEnum):
    Unknown = 0,
    X = 1,
    Y = 2,
    Z = 3, 
    Zrot = 4


class Node:
    
    def __init__(self, ID=None, X=None, Y=None, Z=None):
        self.ID = ID
        self.X = X
        self.Y = Y
        self.Z = Z
        # elements connected to this node
        self.elements_dictionary = {}
        # list of DOFtypes
        self.constraints = []

       
class Element:
    """
    An abstract Element class that defines the basic properties and
    methods of a finite element.
    """
    DOFtypes = None
    
    def __init__(self, ID=None, element_type=None):
        """Creates an element instance.
        
        Parameters
        ----------
        ID : int
            Element ID.
        
        element_type : Element
            Instance of Element with material and thickness only.
        
        Attributes
        ----------
        nodes_dictionary : dict
            Dictionary with nodes' ID as keys and nodes as values.
        DOFtypes : list<list>
            A list with size equal to the number of nodes, each containing
            a list with the degrees of freedom of each node
        DOFs : list
            A list with the DOFs othe element
        DOF_enumerator : GenericDOFEnumerator
        """
       
        self.ID = ID
        self.element_type = element_type
        self.nodes_dictionary = {}
        self.DOF_enumerator = GenericDOFEnumerator()
        
    def add_node(self, node):
        self.nodes_dictionary[node.ID] = node
        
    def add_nodes(self, node_list):
        for node in node_list:
            self.nodes_dictionary.add_node(node)
    
    @property
    def nodes(self):
        return list(self.nodes_dictionary.values())        
    
    @property
    def number_of_nodes(self):
        return len(self.nodes)
    
    @staticmethod
    def get_nodes_for_matrix_assembly(element):
        return element.nodes
    
    @staticmethod
    def get_element_DOFtypes(element):
        return element.DOFtypes
    

#%%
class Model:
    """Model composed of elements and their nodes.
    
    Attributes
    ----------
    nodal_DOFs_dictionary : dict<int, dict<DOFType, int>>
        Dictionary that links node.ID and DOFType with the equivalent 
        global nodal DOF number. 
    loads : list<Load>
        List containing the loads applied to the model nodes.
    forces : np.ndarray<float>
        Force vector applied to model DOFs.
    """

    def __init__(self, nodes_dictionary={}, elements_dictionary={}):
        """Initialize Model instance.
        
        Parameters
        ----------
        nodes_dictionary : dict<int, Node>
            Dictionary of model nodes : { nodeID : node}.
        
        elements_dictionary : dict<int, Element>
            Dictionary of model elements : { elementID : element}.
        """

        self.nodes_dictionary = nodes_dictionary
        self.elements_dictionary = elements_dictionary
        self.nodal_DOFs_dictionary = {}
        self.loads = []
        self.forces = None
        self.global_DOFs = None
    
    @property
    def nodes(self):
        return list(self.nodes_dictionary.values())
    
    @property
    def number_of_nodes(self):
        return len(self.nodes)
    
    @property
    def elements(self):
        return list(self.elements_dictionary.values())
    
    @property
    def number_of_elements(self):
        return len(self.elements)
    
    def build_element_dictionary_of_each_node(self):
        for element in self.elements:
            for node in element.nodes:
                node.elements_dictionary[element.ID] = element

    def enumerate_global_DOFs(self):
        """Enumerates the degrees of freedom of the model."""
       
        # dict e.g. {random_nodeID: [DOFtype.X, DOFtype.Y]}
        nodal_DOFtype_dictionary = {}
        for element in self.elements:
            for i,node in enumerate(element.nodes):
                if node.ID not in nodal_DOFtype_dictionary: #searches in keys
                    # if node not in dictionary, append the node to it
                    nodal_DOFtype_dictionary[node.ID] = element.DOFtypes[i]
       
        total_DOFs = int(0)
        for node in self.nodes:
            # {DOFtype.X: DOFID}
            DOFs_dictionary = {}
            for DOFtype in nodal_DOFtype_dictionary[node.ID]:
                DOF_ID = 0
                if DOFtype in node.constraints:
                    DOF_ID = int(-1)                 
                elif DOF_ID == 0:
                    DOF_ID = int(total_DOFs)
                    total_DOFs += 1
                
                DOFs_dictionary[DOFtype] = DOF_ID               
            
            self.nodal_DOFs_dictionary[node.ID] = DOFs_dictionary
        
        self.total_DOFs = total_DOFs 
         
    def assign_nodal_loads(self):
        """Assigns the loads to the force vector."""
        forces = np.zeros((self.total_DOFs,1))
        for load in self.loads:
            load_global_DOF = self.nodal_DOFs_dictionary[load.node.ID][load.DOF]
            if load_global_DOF>=0:
                forces[load_global_DOF] = load.magnitude
        self.forces = forces 
                
    def connect_data_structures(self):
         self.build_element_dictionary_of_each_node()
         self.enumerate_global_DOFs()
         self.assign_nodal_loads()
    

        
#%%
class Load:
    
    def __init__(self, magnitude=None, node=None, DOF=None):
        self.magnitude = magnitude
        self.node = node
        self.DOF = DOF

#%%
""" 
Gauss integration library
"""

class GaussPoint1D:
    """Defines an one-dimensional Gauss Legendre integration point."""
    
    def __init__(self, coordinate=None, weight=None):
        self.coordinate = coordinate
        self.weight = weight

       
class GaussPoint3D:
    """Defines a three-dimensional Gauss Legendre integration point."""
    
    def __init__(self, ksi, eta, zeta, shape_functions, deformation_matrix, weight):	
        self.ksi = ksi
        self.eta = eta
        self.zeta = zeta
        self.shape_functions = shape_functions
        self.deformation_matrix = deformation_matrix
        self.weight = weight
        

class GaussQuadrature:
    """Provides one-dimensional Gauss-Legendre points and weights."""

    gauss_point1 = GaussPoint1D(coordinate=0, weight=2)
    
    gauss_point2a = GaussPoint1D(coordinate=-.5773502691896, weight=1)    
    gauss_point2b = GaussPoint1D(coordinate=0.5773502691896, weight=1)

    
    @staticmethod
    def get_gauss_points(integration_degree):
#         * For point coordinates, we encounter the following constants:
#         * 0.5773502691896 = 1 / Square Root 3
#         * 0.7745966692415 = (Square Root 15)/ 5
#         * 0.8611363115941 = Square Root( (3 + 2*sqrt(6/5))/7)
#         * 0.3399810435849 = Square Root( (3 - 2*sqrt(6/5))/7)
#         * 
#         * For the weights, we encounter the followings constants:
#         * 0.5555555555556 = 5/9
#         * 0.8888888888889 = 8/9
#         * 0.3478548451375 = (18 - sqrt30)/36
#         * 0.6521451548625 = (18 + sqrt30)/36  
        if integration_degree==1:
            return [GaussQuadrature.gauss_point1]
        elif integration_degree==2:
            return [GaussQuadrature.gauss_point2a,
                    GaussQuadrature.gauss_point2b]
        else:
            raise NotImplementedError("Unsupported degree of integration: {:}".format(integration_degree))
            
            
    






        