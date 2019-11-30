
from fem.core.providers import ElementMassProvider, ElementStiffnessProvider, RayleighDampingMatrixProvider
from fem.core.providers import GlobalMatrixProvider


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


cdef class ProblemStructuralDynamic:
    """Provider responsible for the assembly of the global matrices."""
    cpdef public object model, global_matrix_provider
    cpdef public object _stiffness_matrix, _mass_matrix, _damping_matrix
    cpdef public object stiffness_provider, mass_provider, damping_provider
    cpdef public bint change_stiffness, change_mass, change_damping
    
    def __init__(self, model, global_matrix_provider=GlobalMatrixProvider, damping_provider=None):
        self.model = model
        self.global_matrix_provider = global_matrix_provider
        self._stiffness_matrix = None
        self._mass_matrix = None
        self._damping_matrix = None
        self.stiffness_provider = ElementStiffnessProvider()
        self.mass_provider = ElementMassProvider()
        self.damping_provider = damping_provider
        self.change_stiffness = True
        self.change_mass = True
        self.change_damping = True

    @property
    def stiffness_matrix(self):
        
        if (self._stiffness_matrix is not None) and not self.change_stiffness:
            pass
        elif (self._stiffness_matrix is not None) and self.change_stiffness:
            self.rebuild_stiffness_matrix()
        else:
            self.build_stiffness_matrix()
        return self._stiffness_matrix
    
    @property
    def mass_matrix(self):
        if (self._mass_matrix is not None) and not self.change_mass:
            pass
        elif (self._mass_matrix is not None) and self.change_mass:
            self.rebuild_mass_matrix()
        else:
            self.build_mass_matrix()
        return self._mass_matrix
    
    @property
    def damping_matrix(self):
        if (self._damping_matrix is not None) and not self.change_damping:
            pass
        elif (self._damping_matrix is not None) and self.change_damping:
            self.rebuild_damping_matrix()
        else:
            self.build_damping_matrix()
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
        
    cpdef get_rhs_from_history_load(self, timestep):
        model = self.model
        provider = self.global_matrix_provider
        stforces = model.forces
        dyforces = model.dynamic_forces
        inloads = model.inertia_loads
        in_dir_vectors = model.inertia_forces_direction_vectors 
        return provider.get_rhs_from_history_loads(timestep, stforces, dyforces,
                                                   inloads, in_dir_vectors, self._mass_matrix)
    
    cpdef mass_matrix_vector_product(self, vector):
        return self._mass_matrix @ vector
    
    cpdef stiffness_matrix_vector_product(self, vector):
        return self._stiffness_matrix @ vector
    
    cpdef damping_matrix_vector_product(self, vector):
        return self._damping_matrix @ vector
    
