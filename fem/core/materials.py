

from enum import IntEnum
from dataclasses import dataclass
import numpy as np

class StressState2D(IntEnum):
    plain_stress = 0,
    plain_strain = 1
    
@dataclass
class ElasticMaterial2D:
    young_modulus : np.float64 = None
    poisson_ratio : np.float64 = None
    stress_state : np.float64 = None
    mass_density : np.float64 = 0
    _constitutive_matrix : np.ndarray = None
#    
#    def __init__(self, stress_state=None, young_modulus=None, poisson_ratio=None, mass_density=0):
#        self.young_modulus = young_modulus
#        self.poisson_ratio = poisson_ratio
#        self.stress_state = stress_state
#        self.mass_density = mass_density
#        self._constitutive_matrix = None
        
    
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
        """Given a strain vector updates the material state.
        
        constitutive_matrix =  E/(1-ni*ni) [[1, ni, 0],
                                            [ni, 1, 0],
                                            [0, 0, (1-ni)/2]]
        """
        E = self.young_modulus
        ni = self.poisson_ratio
        constitutive_matrix = np.zeros((3,3))
        
        if self.stress_state==StressState2D.plain_stress:
            
            factor = E/(1-ni*ni)
           
            constitutive_matrix[0, 0] = factor
            constitutive_matrix[1, 1] = factor
            constitutive_matrix[1, 0] = factor * ni
            constitutive_matrix[0, 1] = factor * ni
            constitutive_matrix[2, 2] = factor * (1-ni) * .5
            
        else:
            raise NotImplementedError("PlainStrain not implemented yet.")
        
        self._constitutive_matrix = constitutive_matrix
        if strains is None:
            self.strains = np.zeros((3,1))
            self.stresses = np.zeros((3,1))

        else:
            # σ = E ε
            self.stresses = (constitutive_matrix[:,0] * strains[0] 
                             + constitutive_matrix[:,1] * strains[1] 
                             + constitutive_matrix[:,2] * strains[2])
        

