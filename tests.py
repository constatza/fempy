# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""

import numpy as np





arr = np.zeros((5))

arr[0] = 1
print(arr)

arr[-1] = arr[0]

print(arr)

arr[0] = 0
print(arr)