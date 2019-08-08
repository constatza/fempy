# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:49:31 2019

@author: constatza
"""
from numpy import empty, nanmin, NaN, zeros


class GenericDOFEnumerator:
    """Retrieves element connectivity data required for matrix assembly."""
    
    @staticmethod
    def get_DOF_types(element):
        """Retrieves the dof types of each node."""
        return element.element_type.get_element_DOFtypes(element)
    
    @staticmethod
    def get_DOFtypes_for_DOF_enumeration(element):
        """Retrieves the dof types of each node."""
        return element.element_type.get_element_DOFtypes(element)

    @staticmethod
    def get_nodes_for_matrix_assembly(element):
        """Retrieves the element nodes."""
        return element.nodes

    @staticmethod
    def get_transformed_matrix(matrix):
        """Retrieves matrix transformed from local to global coordinate system."""
        return matrix

    @staticmethod
    def get_tranformed_displacements_vector(vector):
        """ Retrieves displacements transformed from local to global coordinate system."""
        return vector
    
    def get_transformed_forces_vector(vector):
        """Retrieves displacements transformed from local to global coordinate system."""
        return vector

        
class ProblemStructural:
    """Responsible for the assembly of the global stiffness matrix."""
    
    def __init__(self, model):
        self.model = model
        self._matrix = None
        self.stiffness_provider = ElementStructuralStiffnessProvider()

    @property
    def matrix(self):
        if self._matrix is None:
            self.build_matrix()
        else:
            self.rebuild_matrix()
        return self._matrix
    
    def build_matrix(self):
        """ Builds the global Stiffness Matrix"""
        provider = ElementStructuralStiffnessProvider()
        self._matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, provider)
    
    def rebuild_matrix(self):
        """ Rebuilds the global Stiffness Matrix"""
        self._matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, self.stiffness_provider)
    
    def calculate_matrix(self, linear_system):
        if self._matrix == None:
            self.build_matrix()
        linear_system.matrix = self.matrix


class ElementStructuralStiffnessProvider:
    """ 
    Responsible for providing elemental stiffness matrix 
    for the global matrix assembly.
    """
    @staticmethod
    def matrix(element):
        return element.element_type.stiffness_matrix(element)


class GlobalMatrixAssembler:
    """Assembles the global stiffness matrix."""
    
    @staticmethod           
    def calculate_global_matrix(model, element_provider, nodal_DOFs_dictionary=None):
        """Calculates the global stiffness matrix.
       
        Parameters
        ----------
        model : Model
            The model whose matrix is to be calculated.
        nodal_DOFs_dictionary : dict<int, dict<DOFType, int>>
            Dictionary that links node.ID and DOFType with the equivalent 
            global nodal DOF number.
        element_provider : ElementStructuralStiffnessProvider
            The element provider.
        
        Returns
        -------
        global_stiffness_matrix : np.ndarray
            Model global stiffness matrix.
        """
        
        
        if nodal_DOFs_dictionary == None:
            nodal_DOFs_dictionary = model.nodal_DOFs_dictionary
        
        numDOFs = model.total_DOFs
        global_stiffness_matrix = zeros([numDOFs, numDOFs])
        
        for element in model.elements:
            
            element_matrix = element_provider.matrix(element)
            print(element_matrix)
            element_DOFtypes = element.element_type.DOF_enumerator.get_DOF_types(element)
            matrix_assembly_nodes = element.element_type.DOF_enumerator.get_nodes_for_matrix_assembly(element)
            
            element_matrix_row = 0
            for i in range(len(element_DOFtypes)):
                node_row = matrix_assembly_nodes[i]
                for DOFtype_row in element_DOFtypes[i]:
                    DOFrow = nodal_DOFs_dictionary[node_row.ID][DOFtype_row]
                    
                    if DOFrow != -1:
                        
                        element_matrix_column = 0  
                        for j in range(len(element_DOFtypes)):
                            node_column = matrix_assembly_nodes[j]
                            for DOFtype_column in element_DOFtypes[j]:
                                DOFcolumn = nodal_DOFs_dictionary[node_column.ID][DOFtype_column]
                                if DOFcolumn != -1:
                                    global_stiffness_matrix[DOFrow, DOFcolumn] += element_matrix[element_matrix_row,
                                                                                                 element_matrix_column]
                                element_matrix_column += 1
                    
                    element_matrix_row += 1

        return global_stiffness_matrix 
                
            
        