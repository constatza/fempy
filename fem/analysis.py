# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:42:51 2019

@author: constatza
"""
import numpy as np

class MonteCarloMaterialAnalysis:
    
    def __init__(self, analyzer, E, nsims=1):
        self.analyzer = analyzer
        self.nsims = nsims
    
    def run(self, numelX, numelY):
        E = self.E
        nsims = self.nsims
        u = []
        ud = []
        udd = []
#       K = np.empty((self.analyzer.model.total_dofs, nsims))
        total_simulations = E.shape[0]
        for i in range(nsims):
                   
        
            counter = -1
            for i in range(numelX): 
                for j in range(numelY):
                    counter += 1
                    # access elements bottom to top, ascending Y
                    element = model.elements[counter] 
                    element.material.young_modulus = E[i, j]
            
            parent_analyzer.build_matrices()
            parent_analyzer.initialize()
            parent_analyzer.solve()
            
        
    