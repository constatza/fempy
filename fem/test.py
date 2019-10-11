# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy 


from dataclasses import dataclass

@dataclass
class foo():
    a : float = 0

    def nice(self):
        self.a += 1

    @property
    def A(self):
        return "is {:}".format(self.a)
    
    def mice(self):
        nice = self.nice
        nice()
        nice()
        
b = foo()


b.mice()

print(b.A)