# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:49:31 2019

@author: constatza
"""
from numpy import empty


class ProblemStructural:
    """Responsible for the assembly of the global stiffness matrix."""
    
    def __init__(self, model):
        self.model = model
        self._matrix2D = None

    @property
    def matrix2D(self):
        if self._matrix2D == None:
            self.build_matrix()
        else:
            self.rebuild_matrix()
        return self._matrix2D
    
    def build_matrix(self):
        """ Builds the global Stiffness Matrix"""
        provider = ElementStructuralStiffnessProvider()
        self._matrix2D = GlobalMatrixAssembler.CalculateGlobalMatrix(model, provider)
    
    def rebuild_matrix(self):
        """ Rebuilds the global Stiffness Matrix"""
        self._matrix2D = GlobalMatrixAssembler.CalculateGlobalMatrix(model, provider)
    
    def calculate_matrix(self, linear_system):
        if self._matrix2D == None:
            self.build_matrix()
        linear_system.matrix = self.matrix2D


class ElementStructuralStiffnessProvider:
    """ 
    Responsible for providing elemental stiffness matrix 
    for the global matrix assembly.
    """
    def matrix2D(element):
        return element.element_type.stiffness_matrix(element)


class GlobalMatrixAssembler:
    """Assembles the global stiffness matrix."""
    
    @staticmethod
    def calculate_row_index(model, nodal_DOFs_dictionary):
        row_heights = empty(model.total_DOFs)
        
        for element in model.elements:
            for node in element.element_type.DOF_enumerator.
    