

class ProblemStructural:
    """Responsible for the assembly of the global stiffness matrix."""
    
    def __init__(self, model):
        self.model = model
        self._matrix = None
        self.stiffness_provider = ElementStiffnessProvider()
        self.mass_provider = ElementMassProvider()
        self.

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
        self.stiffness_provider = ElementStiffnessProvider()
        self.mass_provider = ElementMassProvider()

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
    
    @stiffness_matrix.setter
    def stiffness_matrix(self, value):
        self._stiffness_matrix = value
    
    @mass_matrix.setter
    def mass_matrix(self, value):
        self._mass_matrix = value
    
    def build_stiffness_matrix(self):
        """ Builds the global Stiffness Matrix"""
        provider = ElementStiffnessProvider()
        self.stiffness_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, provider)
        
    def build_mass_matrix(self):
        """ Builds the global Mass Matrix"""
        provider = ElementMassProvider()
        self.mass_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, provider)
    
    def rebuild_stiffness_matrix(self):
        """ Rebuilds the global Stiffness Matrix"""
        self.stiffness_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, self.stiffness_provider)

    def rebuild_mass_matrix(self):
        """ Rebuilds the global Mass Matrix"""
        self.stiffness_matrix = GlobalMatrixAssembler.calculate_global_matrix(self.model, self.mass_provider)

    def calculate_matrix(self, linear_system):
        linear_system.stiffness_matrix = self.stiffness_matrix
        linear_system.mass_matrix = self.mass_matrix
    