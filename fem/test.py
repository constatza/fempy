# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy.linalg import eigh, eig

a = np.array([[2,2],
              [2,5]])


d = np.sqrt(np.sum(a, axis=1, keepdims=True))

Ptilda = a/d/d.T 
P = 1/d**2 * a

vals1, vecs1 = eig(P)
vals2, vecs2 = eigh(Ptilda)
vecs2 = vecs2