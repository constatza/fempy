

from enum import IntEnum
import numpy as np

class StressState2D(IntEnum):
    plain_stress = 0,
    plain_strain = 1
    

class ElasticMaterial2D:
    
    def __init__(self, stress_state=None, young_modulus=None, poisson_ratio=None):
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.stress_state = stress_state
        self._constitutive_matrix = None
        
    
    @property    
    def constitutive_matrix(self):
        """Gets the constitutive matrix ."""
        
        if self._constitutive_matrix is None:
            self.update_material()
        return self._constitutive_matrix
    
    @constitutive_matrix.setter
    def constitutive_matrix(self, none):
        self._constitutive_matrix = none
        
    
    def update_material(self, strains=None):
        """Given a strain vector updates the material state."""
        if self.stress_state==StressState2D.plain_stress:
            E = self.young_modulus
            ni = self.poisson_ratio
            constitutive_matrix =  E/(1-ni*ni) * np.array([[1, ni, 0],
                                                           [ni, 1, 0],
                                                           [0, 0, (1-ni)/2]])
        else:
            raise NotImplementedError("PlainStrain not implemented yet.")
        
        self._constitutive_matrix = constitutive_matrix
        if strains==None:
            self.strains = np.zeros((3,1))
            self.stresses = np.zeros((3,1))
        else:
            self.stresses = constitutive_matrix @ strains
        

