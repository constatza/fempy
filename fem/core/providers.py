# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:49:31 2019

@author: constatza
"""


from numpy import array, zeros, newaxis, arange, empty
from scipy import sparse
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


