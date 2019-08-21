# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""

import numpy as np

index = np.array([1,3])
index2 = np.array([0,2])
index = index[:,np.newaxis]
z = np.zeros((10,10))

z[index,index2] = 1

z[index[index!=1], index2[index2!=0]] = 999

print(z)