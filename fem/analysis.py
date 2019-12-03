# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:42:51 2019

@author: constatza
"""
import numpy as np

class MonteCarloMaterialAnalysis:
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def run(self, E, numelX, numelY):
        analyzer = self.analyzer
        u = []
        ud = []
        udd = []
        total_simulations = E.shape[0]
        for case in range(1):
            counter = -1
                   
        
            for width in range(numelX):
                for height in range(numelY):
                    #slicing through elements list the geometry rectangle grid is columnwise
                    counter += 1
                    element = analyzer.model.elements[counter] 
                    element.material.young_modulus = E[case, height]
                    print(element.material.young_modulus)        

            parent_analyzer.initialize()
            parent_analyzer.solve()
            
        
    