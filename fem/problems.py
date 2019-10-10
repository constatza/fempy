import numpy as np

from fempy.fem.core.providers import ElementMassProvider, ElementStiffnessProvider
from fempy.fem.assemblers import GlobalMatrixAssembler


class ProblemStructural:
    """Responsible for the assembly of the global stiffness matrix."""
    
    def __init__(self, model):
        self.model = model
        self._matrix = None
        self.stiffness_provider = ElementStiffnessProvider()
        self.mass_provider = ElementMassProvider()


    @property
    def matrix(self):
        if self._matrix is None:
            self.build_matrix()
        else:
            self.rebuild_matrix()
        return self._matrix
    
    def build_matrix(self):
        """ Builds the global Stiffness Matrix"""
        provider = ElementStiffnessProvider()
        self._matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, provider)
 
    def rebuild_matrix(self):
        """ Rebuilds the global Stiffness Matrix"""
        self._matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, self.stiffness_provider)
    
    def calculate_matrix(self, linear_system):
        linear_system.matrix = self.matrix


class ProblemStructuralDynamic:
    """Provider responsible for the assembly of the global matrices."""
    
    def __init__(self, model):
        self.model = model
        self._stiffness_matrix = None
        self._mass_matrix = None
        self._damping_matrix = None
        self.stiffness_provider = ElementStiffnessProvider()
        self.mass_provider = ElementMassProvider()
        self.damping_provider = None

    @property
    def stiffness_matrix(self):
        if self._stiffness_matrix is None:
            self.build_stiffness_matrix()
        else:
            self.rebuild_stiffness_matrix()
        return self._stiffness_matrix
    
    @property
    def mass_matrix(self):
        if self._mass_matrix is None:
            self.build_mass_matrix()
        else:
            self.rebuild_mass_matrix()
        return self._mass_matrix
    
    @property
    def damping_matrix(self):
        if self._damping_matrix is None:
            self.build_damping_matrix()
        else:
            self.rebuild_damping_matrix()
        return self._damping_matrix

    @stiffness_matrix.setter
    def stiffness_matrix(self, value):
        self._stiffness_matrix = value
    
    @mass_matrix.setter
    def mass_matrix(self, value):
        self._mass_matrix = value
    
    @damping_matrix.setter
    def damping_matrix(self, value):
        self._damping_matrix = value
    
    def build_stiffness_matrix(self):
        """ Builds the global Stiffness Matrix"""
        provider = ElementStiffnessProvider()
        self.stiffness_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, provider)
        
    def build_mass_matrix(self):
        """ Builds the global Mass Matrix"""
        provider = ElementMassProvider()
        self.mass_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, provider)
        
    def build_damping_matrix(self):
        """ Builds the global Mass Matrix"""
        
        self.damping_matrix = self._mass_matrix + self._stiffness_matrix


    def rebuild_stiffness_matrix(self):
        """ Rebuilds the global Stiffness Matrix"""
        self.stiffness_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, self.stiffness_provider)

    def rebuild_mass_matrix(self):
        """ Rebuilds the global Mass Matrix"""
        self.mass_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, self.mass_provider)
    
    def rebuild_damping_matrix(self):
        """ Rebuilds the global Mass Matrix"""
        self.damping_matrix = self.mass_matrix + self.build_stiffness_matrix

    def get_rhs_from_load_history(self, timestep):
        static_forces = self.model.forces 
        dynamic_forces = np.zeros(static_forces.shape)
        for dof, history in self.model.dynamic_forces.items():
            dynamic_forces[dof] = history[timestep]
        
        rhs = static_forces + dynamic_forces
        
        return rhs
    
    def linear_combination_into_stiffness(self, coeffs):
        K_eff = (coeffs['stiffness'] * self.stiffness_matrix 
                + coeffs['mass'] * self.mass_matrix
                + coeffs['damping'] * self.damping_matrix)
        return K_eff
    
    def mass_matrix_vector_product(self, vector):
        return self._mass_matrix @ vector
    
    def stiffness_matrix_vector_product(self, vector):
        return self._stiffness_matrix @ vector
    
    def damping_matrix_vector_product(self, vector):
        return self._damping_matrix @ vector
    
