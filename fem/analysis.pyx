# cython: language_level=3

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:42:51 2019

@author: constatza
"""
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef material_monte_carlo(object analyzer, np.ndarray[np.float64_t, ndim=2] E, 
         int numelX, int numelY):
    cdef object model = analyzer.model
#    u = []

#    total_simulations = E.shape[0]
    cdef size_t case, width, height
    for case in range(E.shape[0]):
        counter = -1

        for width in range(numelX):
            for height in range(numelY):
                #slicing through elements list the geometry rectangle grid is columnwise
                counter += 1
                element = model.elements[counter] 
                element.material.young_modulus = E[case, height]
                print(element.material.young_modulus)        

        analyzer.initialize()
        analyzer.solve()
            
        
    