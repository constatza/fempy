

import numpy as np
from scipy.sparse import csr_matrix
from mathematics.manilearn import matrix_density


class Analyzer:
    """Abstract class for Parent Analyzers"""
        
    def __init__(self, provider=None, child_analyzer=None):
        """
        Creates an instance that uses a specific problem type and an
        appropriate child analyzer for the construction of the system of 
        equations arising from the actual physical problem.
        
        Parameters
        ----------
        provider : ProblemType
            Instance of the problem type to be solved.
        child_analyzer : Analyzer
            Instance of the child analyzer that will handle the solution of
            the system of equations.
        """
        self.provider = provider    
        self.child = child_analyzer
        self.child.parent = self       

    
class Linear:
    """ 
    This class makes the appropriate arrangements 
    for the solution of linear systems of equations.
    """

    def __init__(self, solver=None):
        """
        Initializes an instance of the class.
        
        Parameters
        ----------
        solver : Solver
            The solver instance that will solve the linear system of equations.
        
        Attributes
        ----------
        parent : Analyzer
            Instance of the child analyzer that will handle the solution of
            the system of equations.
        """
        # 
        self.solver = solver
        # The parent analyzer that transforms the physical problem
        # to a system of equations
        self.parent = None
    
    def initialize(self):
        """ 
        Makes the proper solver-specific initializations before the solution
        of the linear system of equations. This method MUST be called before 
        the actual solution of the system.
        """
        self.solver.initialize()

    def solve(self):
        """
        Solves the linear system of equations by calling the corresponding 
        method of the specific solver attached during construction of 
        current instance
        """
        self.solver.solve()


class Static(Analyzer):
    """
    This class constructs the system of equations to be solved and utilizes 
    a child analyzer for handling the solution of these equations.
    e.g. For static, linear analysis we have the relation:  
    StaticAnalyzer.child =  LinearAnalyzer
    """
    
    def __init__(self, provider, child_analyzer, linear_system):
        """
        Creates an instance that uses a specific problem type and an
        appropriate child analyzer for the construction of the system of 
        equations arising from the actual physical problem.
        
        Parameters
        ----------
        provider : ProblemType
            Instance of the problem type to be solved.
        child_analyzer : Analyzer
            Instance of the child analyzer that will handle the solution of
            the system of equations.
        linear_system 
            Instance of the linear system that will be initialized.
        """
        super().__init__(provider, child_analyzer)    
        self.linear_system = linear_system
 
    def build_matrices(self):
        """
        Builds the appropriate linear system matrix and updates the 
        linear system instance used in the constructor.
        """
        self.provider.calculate_matrix(self.linear_system)

    def initialize(self):
        """
        Makes the proper solver-specific initializations before the solution 
        of the linear system of equations. This method MUST be called BEFORE 
        the actual solution of the aforementioned system 
        """
        if self.child==None:
            raise ValueError("Static analyzer must contain a child analyzer.")
        self.child.initialize()

    def solve(self):
        """
        Solves the linear system of equations by calling the corresponding 
        method of the specific solver of current instance.
        """
        if self.child==None:
            raise ValueError("Static analyzer must contain a child analyzer.")
        self.child.solve()


class NewmarkDynamicAnalyzer(Analyzer):
    """Implements the Newmark method for dynamic analysis."""

    
    def __init__(self, model=None, solver=None, provider=None, child_analyzer=None, 
                 timestep=None, total_time=None, acceleration_scheme='constant'):

        super().__init__(provider, child_analyzer)
        self.model = model
        self.solver = solver
        self.timestep = timestep
        self.total_time = total_time
        self.total_steps = int(total_time/timestep)
        self.set_scheme(acceleration_scheme)
        self.calculate_coefficients()
        self.linear_system = solver.linear_system
        self.rhs = None
        self.u = None
        self.ud = None
        self.udd = None

    def set_scheme(self, acceleration_scheme):
        if acceleration_scheme=='constant':
            self.delta=1/2
            self.alpha=1/4
    
    def calculate_coefficients(self):
        alpha = self.alpha
        delta = self.delta
        timestep = self.timestep
        alphas = np.empty(8, dtype=np.float64)
        alphas[0] = 1 / (alpha * timestep * timestep)
        alphas[1] = delta / (alpha * timestep)
        alphas[2]= 1 / (alpha * timestep)
        alphas[3] = 1 / (2 * alpha) - 1
        alphas[4] = delta/alpha - 1
        alphas[5] = timestep * 0.5 * (delta/alpha - 2)
        alphas[6] = timestep * (1 - delta)
        alphas[7] = delta * timestep
        self.alphas = alphas
    
    def build_matrices(self):
        """
        Makes the proper solver-specific initializations before the 
        solution of the linear system of equations. This method MUST be called 
        before the actual solution of the aforementioned system
        """
        a0 = self.alphas[0]
        a1 = self.alphas[1]
        
        self.linear_system.matrix = (self.stiffness_matrix 
                                    + a0 * self.mass_matrix
                                    + a1 * self.damping_matrix)


    def initialize_internal_vectors(self, u0=None, ud0=None):
        if self.linear_system.solution is not None:
            self.linear_system.reset()
            
        provider = self.provider
        stiffness = np.ascontiguousarray(provider.stiffness_matrix.astype(float))
        mass = np.ascontiguousarray(provider.mass_matrix.astype(float))
        damping = np.ascontiguousarray(provider.damping_matrix.astype(float)) #after M and K !
        
        density = matrix_density(damping)
        if density < 0.9:
            print('Using sparse Linear Algebra')
            mass = csr_matrix(mass)
            stiffness = csr_matrix(stiffness)
            damping = csr_matrix(damping)
        
        total_dofs = stiffness.shape[0]
        self.displacements = np.empty((total_dofs, self.total_steps), dtype=np.float32)
        self.velocities = np.empty((total_dofs, self.total_steps), dtype=np.float32)
        self.accelerations = np.empty((total_dofs, self.total_steps), dtype=np.float32)

        if u0 is None:
            u0 = np.zeros(total_dofs)            
        if ud0 is None:
            ud0 = np.zeros(total_dofs)
        
        provider.calculate_inertia_vectors() # before first call of get_rhs...
        rhs0 = self.provider.get_rhs(0)
        self.linear_system.rhs = rhs0 - stiffness.dot(u0) - damping.dot(ud0)
        self.linear_system.matrix = mass
        self.solver.initialize()
        self.solver.solve()
        self.udd = self.linear_system.solution
        self.ud = ud0
        self.u = u0
        self.store_results(0)
        self.mass_matrix = mass
        self.stiffness_matrix = stiffness
        self.damping_matrix = damping
        self.linear_system.reset()

    def initialize(self):
        """
        Initializes the models, the solvers, child analyzers, builds
        the matrices, assigns loads and initializes right-hand-side vectors.
        """
        linear_system = self.linear_system
        model = self.model
        
        model.connect_data_structures()

        linear_system.reset()

        model.assign_loads()
        
        self.initialize_internal_vectors() # call BEFORE build_matrices & initialize_rhs
        self.build_matrices()
     
        self.linear_system.rhs = self.provider.get_rhs(1)
        
        self.child.initialize()
    
        
   
    def solve(self):
        """
        Solves the linear system of equations by calling the corresponding 
        method of the specific solver attached during construction of the
        current instance.
        """
        # initialize functions to adef self.function() overhead
        get_rhs = self.provider.get_rhs
        calculate_rhs_implicit = self.calculate_rhs_implicit
        child_solve = self.child.solve
        update_velocity_and_acceleration = self.update_velocity_and_accelaration
        store_results = self.store_results

        for i in range(1, self.total_steps):
            
            self.rhs = get_rhs(i)
            
            self.linear_system.rhs = calculate_rhs_implicit()            
            child_solve()
            
            update_velocity_and_acceleration()
            store_results(i)
    
    def calculate_rhs_implicit(self):
        """
        Calculates the right-hand-side of the implicit dynamic method. 
        This will be used for the solution of the linear dynamic system.
        """
        alphas = self.alphas
        u = self.u
        ud = self.ud
        udd = self.udd

        udd_eff = alphas[0] * u + alphas[2] * ud + alphas[3] * udd
        ud_eff = alphas[1] * u + alphas[4] * ud + alphas[5] * udd
      
        inertia_forces = self.mass_matrix.dot(udd_eff)
        damping_forces = self.damping_matrix.dot(ud_eff)
        rhs_effective = inertia_forces + damping_forces + self.rhs   

        return rhs_effective


    def update_velocity_and_accelaration(self):
        
        udd = self.udd
        ud = self.ud 
        u = self.u
      
        u_next = self.linear_system.solution
  
        udd_next = self.alphas[0] * (u_next - u) - self.alphas[2] * ud - self.alphas[3] * udd 
        ud_next = ud + self.alphas[6] * udd +  self.alphas[7] * udd_next

        self.u = u_next
        self.ud = ud_next
        self.udd = udd_next
    

    def store_results(self, timestep):
        self.displacements[:, timestep] = self.u.astype(float)
        self.velocities[:, timestep] = self.ud.astype(float)
        self.accelerations[:, timestep] = self.udd.astype(float)
        
        

