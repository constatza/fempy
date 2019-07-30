# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:49:31 2019

@author: constatza
"""
from numpy import empty, nanmin, NaN


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
    
    # POSSIBLY USELESS
    @staticmethod
    def calculate_row_index(model, nodal_DOFs_dictionary):
        """Calculates row indices for the assembly of global stiffness matrix.
        
        Parameters
        ----------
        nodal_DOFs_dictionary : dict<int, dict<DOFType, int>>
            Dictionary that links node.ID and DOFType with the equivalent 
            global nodal DOF number.
        """

        row_heights = empty(model.total_DOFs)
        minDOF = NaN
        for element in model.elements:
            
            for node in element.element_type.DOF_enumerator.get_nodes_for_matrix_assembly(element):
                
                for dof in nodal_DOFs_dictionary[node.ID].values():
                    
                    if dof != -1:
                        minDOF = nanmin(minDOF, dof)
                    
    