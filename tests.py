# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""

import numba as nb
import numpy as np



 
class A:
    a = 1

   
    def do(self):
        print(self.a)


class foo(A):
    a = 2

print(foo().a)