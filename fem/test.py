# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from numba import cuda
import numpy as np
from fempy.mathematics.manilearn import LinearMap
from dataclasses import dataclass
import math


R = np.random.rand(3,3)

r = np.random.rand(2,3)



@cuda.jit
def increment(an_array):
    pos = cuda.grid(1)
    if pos < an_array.size:
        an_array[pos] += 1

an_array = np.ones(1000000000)
threadsperblock = (16,)
blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
#blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x,)
increment[blockspergrid, threadsperblock](an_array)
   