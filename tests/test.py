# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import pyximport
#pyximport.install()
import fempy.mathematics.linalg as linalg
from fempy.fem.problems import ProblemStructuralDynamic
import timeit 
import numpy as np
R = np.random.rand(3,3)

r = np.random.rand(2,3)


#

#
#
#import cython_module as cm
#
#myclass = cm.CyClass(1000, cm.PyClass())
#myclass.boo()
#
#child = cm.Child(10, cm.PyClass())

a = np.ones(10)
dot = linalg.dot(a,a)


print("cython is {:.2f} times faster".format(py/cy))