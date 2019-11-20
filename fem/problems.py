import numpy as np
#from numba import cuda
from numba import guvectorize, jit, float64, void
from fempy.fem.core.providers import ElementMassProvider, ElementStiffnessProvider, RayleighDampingMatrixProvider
from fempy.fem.core.providers import GlobalMatrixProvider


class ProblemStructural:
    """Responsible for the assembly of the global stiffness matrix."""
    
    def __init__(self, model, global_matrix_provider=None):
        self.model = model
        self._matrix = None
        self.stiffness_provider = ElementStiffnessProvider()
        self.mass_provider = ElementMassProvider()
        if global_matrix_provider is None:
            self.global_matrix_provider = GlobalMatrixProvider(self.model, self.stiffness)
        else:
            self.global_matrix_provider = global_matrix_provider

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
        global_provider = self.global_matrix_provider
        self._matrix = global_provider.get_stiffness_matrix(self.model, provider)
 
    def rebuild_matrix(self):
        """ Rebuilds the global Stiffness Matrix"""
        global_provider = self.global_matrix_provider
        self._matrix = global_provider.get_stiffness_matrix(self.model, self.provider)
    
    def calculate_matrix(self, linear_system):
        linear_system.matrix = self.matrix


class ProblemStructuralDynamic:
    """Provider responsible for the assembly of the global matrices."""
    
    def __init__(self, model, global_matrix_provider=GlobalMatrixProvider, damping_provider=None):
        self.model = model
        self.global_matrix_provider = global_matrix_provider
        self._stiffness_matrix = None
        self._mass_matrix = None
        self._damping_matrix = None
        self.stiffness_provider = ElementStiffnessProvider()
        self.mass_provider = ElementMassProvider()
        self.damping_provider = damping_provider

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
        global_provider = self.global_matrix_provider
        self.stiffness_matrix = global_provider.get_stiffness_matrix(self.model, provider)
        
    def build_mass_matrix(self):
        """ Builds the global Mass Matrix"""
        provider = ElementMassProvider()
        global_provider = self.global_matrix_provider
        self.mass_matrix = global_provider.get_mass_matrix(self.model, provider)
        
    
    def build_damping_matrix(self):
        """ Builds the global Mass Matrix"""
        global_provider = self.global_matrix_provider
        self.damping_matrix = global_provider.get_damping_matrix(self._stiffness_matrix,
                                                               self._mass_matrix, 
                                                               self.damping_provider)


    def rebuild_stiffness_matrix(self):
        """ Rebuilds the global Stiffness Matrix"""
        global_provider = self.global_matrix_provider
        self.stiffness_matrix = global_provider.get_stiffness_matrix(self.model, self.stiffness_provider)

    def rebuild_mass_matrix(self):
        """ Rebuilds the global Mass Matrix"""
        global_provider = self.global_matrix_provider
        self.mass_matrix = global_provider.get_mass_matrix(self.model, self.mass_provider)
    
    def rebuild_damping_matrix(self):
        """ Rebuilds the global Mass Matrix"""
        global_provider = self.global_matrix_provider
        self.damping_matrix = global_provider.get_damping_matrix(self._stiffness_matrix,
                                                               self._mass_matrix,
                                                               self.damping_provider)
        
    def get_rhs_from_history_load(self, timestep):
        provider = self.global_matrix_provider
        stforces = self.model.forces
        dyforces = self.model.dynamic_forces
        return provider.get_rhs_from_history_load(timestep, stforces, dyforces)
    
    def mass_matrix_vector_product(self, vector):
        return self._mass_matrix @ vector
    
    def stiffness_matrix_vector_product(self, vector):
        return self._stiffness_matrix @ vector
    
    def damping_matrix_vector_product(self, vector):
        return self._damping_matrix @ vector
    
#@guvectorize([void(float64[:,:], float64[:,:], float64[:,:])], '(m,l),(l,n)->(m,n)', target='cuda')
#def matmul_gu3(A, B, out):
#    """Perform square matrix multiplication of out = A * B
#    """
#    i, j = cuda.grid(2)
#    if i < out.shape[0] and j < out.shape[1]:
#        tmp = 0.
#        for k in range(A.shape[1]):
#            tmp += A[i, k] * B[k, j]
#        out[i, j] = tmp