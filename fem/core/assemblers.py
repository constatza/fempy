# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:49:31 2019

@author: constatza
"""


from numpy import array, zeros, newaxis, arange, empty
from scipy import sparse
from numba import njit, prange, parfor


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
        self.stiffness_provider = ElementStiffnessProvider()
        self.mass_provider = ElementMassProvider()

    @property
    def matrix(self):
        if self._matrix is None:
            self.build_matrix()
        else:
            self.rebuild_matrix()
        return self._matrix
    
    def build_matrix(self):
        """ Builds the global Stiffness Matrix"""
        provider = ElementStiffnessProvider()
        self._matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, provider)
    
    def rebuild_matrix(self):
        """ Rebuilds the global Stiffness Matrix"""
        self._matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, self.stiffness_provider)
    
    def calculate_matrix(self, linear_system):
        linear_system.matrix = self.matrix


class ProblemStructuralDynamic:
    """Responsible for the assembly of the global stiffness matrix."""
    
    def __init__(self, model):
        self.model = model
        self._stiffness_matrix = None
        self._mass_matrix = None
        self.stiffness_provider = ElementStiffnessProvider()
        self.mass_provider = ElementMassProvider()

    @property
    def stiffness_matrix(self):
        if self._stiffness_matrix is None:
            self.build_stiffness_matrix()
        else:
            self.rebuild_stiffness_matrix()
        return self._stiffness_matrix
    
    @property
    def mass_matrix(self):
        if self._stiffness_matrix is None:
            self.build_mass_matrix()
        else:
            self.rebuild_mass_matrix()
        return self._mass_matrix
    
    @stiffness_matrix.setter
    def stiffness_matrix(self, value):
        self._stiffness_matrix = value
    
    @mass_matrix.setter
    def mass_matrix(self, value):
        self._mass_matrix = value
    
    def build_stiffness_matrix(self):
        """ Builds the global Stiffness Matrix"""
        provider = ElementStiffnessProvider()
        self.stiffness_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, provider)
        
    def build_mass_matrix(self):
        """ Builds the global Mass Matrix"""
        provider = ElementMassProvider()
        self.mass_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, provider)
    
    def rebuild_stiffness_matrix(self):
        """ Rebuilds the global Stiffness Matrix"""
        self.stiffness_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, self.stiffness_provider)

    def rebuild_mass_matrix(self):
        """ Rebuilds the global Mass Matrix"""
        self.stiffness_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, self.mass_provider)

    def calculate_matrix(self, linear_system):
        linear_system.stiffness_matrix = self.stiffness_matrix
        linear_system.mass_matrix = self.mass_matrix


class ElementStiffnessProvider:
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


class ElementMassProvider:
    """ Responsible for providing elemental mass matrix 
    for the global matrix assembly.
        
        Deploys the fact that only the material has changed.
    """
    @staticmethod
    def matrix(element):
        return element.element_type.mass_matrix(element)


class GlobalMatrixAssembler:
    """Assembles the global stiffness matrix."""
    
    @staticmethod           
    def calculate_global_matrix(model, element_provider, nodal_DOFs_dictionary=None):
        """Calculates the generic global matrix. The type of matrix i.e. stiffness,
        mass etc. is defined by the type of the element_provider."
       
        Parameters
        ----------
        model : Model
            The model whose matrix is to be calculated.
        nodal_DOFs_dictionary : dict<int, dict<DOFType, int>>
            Dictionary that links node.ID and DOFType with the equivalent 
            global nodal DOF number.
        element_provider : ElementStiffnessProvider
            The element provider.
        
        Returns
        -------
        global_stiffness_matrix : np.ndarray
            Model global stiffness matrix.
        """
        
        
        if nodal_DOFs_dictionary is None:
            nodal_DOFs_dictionary = model.nodal_DOFs_dictionary
        
        numels = model.number_of_elements
        globalDOFs = empty((numels, 8), dtype=int)
        total_element_matrices = empty((8, 8, numels))
                
        #!!!!! Change if using different type of elements
        element = model.elements[0]
        get_DOF_types = element.element_type.DOF_enumerator.get_DOF_types
        get_nodes_for_matrix_assembly = element.element_type.DOF_enumerator.get_nodes_for_matrix_assembly
        for k,element in enumerate(model.elements):
            
            total_element_matrices[:, :, k] = element_provider.matrix(element)
            
            if model.global_DOFs is None: 
                element_DOFtypes = get_DOF_types(element)
                matrix_assembly_nodes = get_nodes_for_matrix_assembly(element)
                
                counter = -1
                for i in range(len(element_DOFtypes)):
                    node = matrix_assembly_nodes[i]
                    for DOFtype in element_DOFtypes[i]:
                        counter += 1
                        globalDOFs[k, counter] = nodal_DOFs_dictionary[node.ID][DOFtype]
                        
            else:
                globalDOFs = model.global_DOFs

        numDOFs = model.total_DOFs
        globalDOFs = globalDOFs.astype(int)
        global_stiffness_matrix = GlobalMatrixAssembler.assign_element_to_global_matrix(
                                                                        total_element_matrices,
                                                                        globalDOFs,
                                                                        numDOFs)                                                           
        model.global_DOFs = globalDOFs
        return global_stiffness_matrix 
                
    @staticmethod
    @njit('float64[:, :](float64[:, :, :], int32[:, :], int64)')
    def assign_element_to_global_matrix(element_matrices, globalDOFs, numDOFs):
        K = zeros((numDOFs, numDOFs))
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