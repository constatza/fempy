# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import timeit 
import numpy as np
R = np.random.rand(3,3)

r = np.random.rand(2,3)


cy = timeit.timeit('cython_module.foo(1000)', setup='import cython_module', number=1000)
py = timeit.timeit('python_module.foo(1000)', setup='import python_module', number=1000)

print("cython is {:.2f} times faster".format(py/cy))