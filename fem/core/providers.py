# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:49:31 2019

@author: constatza
"""

from fempy.fem.assemblers import GlobalMatrixAssembler
from numpy import array, zeros, newaxis, arange, empty, sqrt
import scipy.linalg as linalg
from numba import njit, prange, parfor


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



class GlobalMatrixProvider:
    
    def get_rhs_from_history_load(timestep, static_forces, dynamic_forces):
        dynamic_forces_vector = zeros(static_forces.shape)
        for dof, history in dynamic_forces.items():
            dynamic_forces_vector[dof] = history(timestep)
        
        return static_forces + dynamic_forces_vector
    
    def get_mass_matrix(model, element_mass_provider):
        mass = GlobalMatrixAssembler.calculate_global_matrix(model, element_mass_provider)
        return mass
        
    def get_stiffness_matrix(model, element_stiffness_provider):
        stiff = GlobalMatrixAssembler.calculate_global_matrix(model, element_stiffness_provider)
        return stiff
    
    def get_damping_matrix(K, M, damping_provider):
        if damping_provider is None:
            damp = zeros(K.shape)
        else:
            damp = damping_provider.calculate_global_matrix(K, M)
        return damp


class ReducedGlobalMatrixProvider(GlobalMatrixProvider):
    
    def __init__(self, linear_map):
        self.linear_map = linear_map
        self.get_mass_matrix = self.reduced_matrix(super().get_mass_matrix)
        self.get_stiffness_matrix = self.reduced_matrix(super().get_stiffness_matrix)
        self.get_damping_matrix = self.reduced_matrix(super().get_damping_matrix)
        self.get_rhs_fron_history_load = self.reduced_vector(super().get_rhs_fron_history_load)
        
        
    
  
    def reduced_matrix(self, func):
        def wrapper(*args, **kwargs):
            matrix = func(*args, **kwargs)
            return self.linear_map.direct_transform_matrix(matrix)
        return wrapper
    
    def reduced_vector(self, func):
        def wrapper(*args, **kwargs):
            vector = func(*args, **kwargs)
            return self.linear_map.direct_transform_vector(vector)
        return wrapper
    
#    def get_mass_matrix(self, model, element_mass_provider):
#        mass = GlobalMatrixAssembler.calculate_global_matrix(model, 
#                                                             element_mass_provider)
#        return self.linear_map.direct_transform_matrix(mass)
#    
#    def get_stiffness_matrix(self, model, element_stiffness_provider):
#        stiff = GlobalMatrixAssembler.calculate_global_matrix(model, 
#                                                              element_stiffness_provider)
#        return self.linear_map.direct_transform_matrix(stiff)
#    
#    def get_damping_matrix(self, K, M, damping_provider):
#        damp = damping_provider.calculate_global_matrix(K, M)
#        return self.linear_map.direct_transform_matrix(damp)
#   
    
    
    

class RayleighDampingMatrixProvider:
    
    def __init__(self, coeffs=None):
        self.coeffs = coeffs
    
    def calculate_global_matrix(self, stiffness_matrix, mass_matrix):
        damping_coeffs = self.coeffs
        eigvals = linalg.eigh(stiffness_matrix, b=mass_matrix,
                                 eigvals=(0,1), eigvals_only=True)

        wmegas = sqrt(eigvals)
        self.frequencies = wmegas
        Matrix = .5* array([1/wmegas,wmegas]) 
        a = linalg.solve(Matrix.T, damping_coeffs)
        
        return a[0]*mass_matrix + a[1]*stiffness_matrix
    
     