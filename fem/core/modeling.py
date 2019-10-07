# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:44:21 2019

@author: constatza
"""
import numpy as np
from enum import IntEnum
from .assemblers import GenericDOFEnumerator


    

        
#%%
class Load:
    
    def __init__(self, magnitude=None, node=None, DOF=None):
        self.magnitude = magnitude
        self.node = node
        self.DOF = DOF


class TimeHistory(Load):
    
    def __init__(self, time_history=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.magnitude is None: self.magnitude = 1
        self.history = self.magnitude * time_history
    
    def get_current_
  
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
            
            
    






        