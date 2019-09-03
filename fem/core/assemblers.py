# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:49:31 2019

@author: constatza
"""


from numpy import array, zeros, newaxis, arange, empty
from scipy import sparse
from numba import njit

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
        linear_system.matrix = self.matrix
        
        

class ElementStructuralStiffnessProvider:
    """ Responsible for providing elemental stiffness matrix 
    for the global matrix assembly.
    """
    @staticmethod
    def matrix(element):
        return element.element_type.stiffness_matrix(element)

class ElementMaterialOnlyStiffnessProvider:
    """ Responsible for providing elemental stiffness matrix 
    for the global matrix assembly.
        
        Deploys the fact that only the material has changed.
    """
    @staticmethod
    def matrix(element):
        points = element.integration_points
        thickness = element.thickness
        element.material.constitutive_matrix = None
        materials = element.materials_at_gauss_points
        return element.calculate_stiffness_matrix(points, materials, thickness)
    
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
        
        
        if nodal_DOFs_dictionary is None:
            nodal_DOFs_dictionary = model.nodal_DOFs_dictionary
        
        numels = model.number_of_elements
        globalDOFs = empty((numels, 8))
        total_element_matrices = empty((8, 8, numels))
        for k,element in enumerate(model.elements):
            
            total_element_matrices[:, :, k] = element_provider.matrix(element)
            element_DOFtypes = element.element_type.DOF_enumerator.get_DOF_types(element)
            matrix_assembly_nodes = element.element_type.DOF_enumerator.get_nodes_for_matrix_assembly(element)
            
            counter = -1
            for i in range(len(element_DOFtypes)):
                node = matrix_assembly_nodes[i]
                for DOFtype in element_DOFtypes[i]:
                    counter += 1
                    globalDOFs[k,counter] = nodal_DOFs_dictionary[node.ID][DOFtype]
                    
        numDOFs = model.total_DOFs            
        globalDOFs = globalDOFs.astype(int)
        global_stiffness_matrix = zeros((numDOFs, numDOFs))
        global_stiffness_matrix = GlobalMatrixAssembler.assign_element_to_global_matrix(
                                                                        global_stiffness_matrix,
                                                                        total_element_matrices,
                                                                        globalDOFs)
                                                                  
                                                                  
        return global_stiffness_matrix 
                
    @staticmethod
    @njit('float64[:, :](float64[:, :], float64[:, :, :], int32[:, :])')
    def assign_element_to_global_matrix(K, element_matrices, globalDOFs):
         
        for ielement in range(element_matrices.shape[2]):
            for i in range(8):
                DOFrow = globalDOFs[ielement, i]
                if DOFrow != -1:
                    for j in range(i, 8):
                        DOFcol = globalDOFs[ielement, j]
                        if DOFcol != -1:
                            K[DOFrow, DOFcol] += element_matrices[i, j, ielement]
                            K[DOFcol, DOFrow] = K[DOFrow, DOFcol]
         
                            
        return K