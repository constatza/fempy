# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""

import numba as nb
import numpy as np





def boo():
    b = np.zeros((2,3))
    c = b[:,0]
    print(c[:])
    
boo()