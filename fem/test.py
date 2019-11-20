# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from fempy.mathematics.manilearn import LinearMap
from dataclasses import dataclass



R = np.random.rand(3,3)

r = np.random.rand(2,3)


a = LinearMap(domain=R, codomain=r)

@dataclass
class full:
    data : float
    
    def get_data(self):
        return self.data

@dataclass
class reduced(full):
    lmap : float
    
    def __post_init__(self):
        self.get_data = self.reduce(super().get_data)
        
    def reduce(self, func):
        
        def wrapper(*args, **kwargs):
            a = func(*args, **kwargs)
            return self.lmap * a
        return wrapper
    
 
a = full(10)
b = reduced(data=10, lmap=0.5)




