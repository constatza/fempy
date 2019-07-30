# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:40:03 2019

@author: constatza
"""
import numpy as np
import numba as nb



def elasticity_matrix_plain_stress(E, ni):

        D = E/(1-ni*ni) * np.array([[1, ni, 0],
                                    [ni, 1, 0],
                                    [0, 0, (1-ni)/2]])
        return D


def get_stiffness_matrix(XY, thickness=0, young_modulus=0, poisson_ratio=0):
        D = elasticity_matrix_plain_stress(young_modulus, poisson_ratio)
        t = thickness       
        return calculate_stiffness_matrix(XY, D, t)
    
    
@nb.njit
def calculate_stiffness_matrix(XY, D, t):      
    k = np.zeros((8, 8))
    #gauss_points3 = np.array([-0.77459667,  0.        ,  0.77459667])
    #gauss_weights3 = np.array([0.55555556, 0.88888889, 0.55555556])
    gauss_points = np.array([-.57735, .57735])
    ns =  range(len(gauss_points))
    for i in ns:
        for j in ns:           
            g = gauss_points[i]
            h = gauss_points[j]
                     
            Dn = .25*np.array([[-1+h, 1-h, 1+h, -1-h],
                               [-1+g, -1-g, 1+g, 1-g]])
        
            J =  Dn @ XY 
            
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    

            B1 = 1/detJ * np.array([[J[1, 1], -J[0, 1],       0,         0],
                                    [       0,       0, -J[1, 0],   J[0, 0]],
                                    [-J[1, 0],  J[0, 0],  J[1, 1], -J[0, 1]]])

            B2 = .25*np.array([[-1+h, 0, 1-h, 0, 1+h, 0, -1-h, 0],
                               [-1+g, 0, -1-g, 0, 1+g, 0, 1-g, 0],
                               [0, -1+h, 0, 1-h, 0, 1+h, 0, -1-h],
                               [0, -1+g, 0, -1-g, 0, 1+g, 0, 1-g]])
                    
            B = B1 @ B2
            
            dk = B.T @ D @ B * detJ * t 
           
            k +=  dk

    return k
        

