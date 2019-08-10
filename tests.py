# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""

import numba as nb
import numpy as np





def boo():
    a = [1,2,3]
    b = np.array([1,2,3,4])
    print(b[a])
    
boo()