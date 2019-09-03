# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 14:59:33 2019

@author: constatza
"""
from copy import deepcopy

a = { 1 : {1 : 1} , 2 :{ 2 : 2}}

outer = a.keys()

inner = next(iter(a.values()))


b = dict.fromkeys(outer, None)

b


