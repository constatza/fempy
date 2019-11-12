# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 17:34:53 2019

@author: constatza
"""

import numpy as np
from dataclasses import dataclass, field
import scipy.linalg as linalg


@dataclass
class Map:
    domain_data : float
    codomain_data : np.ndarray = field(default_factory=np.ndarray)


@dataclass
class LinearMap(Map):
    matrix : np.ndarray = field(init=False)
    
    def __post_init__(self):
        
         p, res, rnk, s = linalg.lstsq(self.domain_data.T, self.codomain_data.T)
         self.matrix = p.T
         self.residuals = res
         self.rank = rnk
         self.singular_values = s
        
    
    def inverse_matrix(self) -> np.ndarray:
        matrix = self.matrix
        try: 
            return linalg.inv(matrix)
        except ValueError:
            print("""Warning: returning pseudoinverse!""")
            return linalg.inv(matrix.T @ matrix)
    
    def direct_transform(self, vector: np.ndarray) -> np.ndarray:
        return self.matrix @ vector
    
    def inverse_tranform(self, vector: np.ndarray, *args, **kwargs) -> np.ndarray:
        matrix = self.matrix 
        return linalg.solve( matrix.T @ matrix, matrix.T @ vector)


def znormalized(array, axis=-1, no_return=True):
    mean = np.atleast_1d(np.mean(array, axis=axis))

    std =  np.atleast_1d(np.std(array, axis=axis))
    std[std==0] = 1
    norm_array = (array - np.expand_dims(mean, axis=axis)) / np.expand_dims(std, axis=axis)
    if no_return:
        return norm_array
    else:
        return norm_array, mean, std


def zdenormalized(array, mean, std):
    return array*std + mean
        
