

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


class Analyzer(ABC):
    """Abstract class for Parent Analyzers"""
    
    @abstractmethod
    def __init__(self, provider, child_analyzer):
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

 
    def build_matrices(self):
        """
        Builds the appropriate system matrix and updates the system instance 
        used in the constructor.
        """
        pass
    
    def initialize(self, is_first_analysis=False):
        pass
    
    def solve(self):
        pass
    

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
    
    def __init__(self, model=None, solver=None, provider=None, child_analyzer=None, timestep=None, total_time=None, alpha=None, delta=None):
        
        super().__init__(provider, child_analyzer)
        self.model = model
        self.solver = solver
        self.timestep = timestep
        self.total_time = total_time
        self.total_steps = int(total_time/timestep)
        self.alpha = alpha
        self.delta = delta
        self.calculate_coefficients()
        self.linear_system = solver.linear_system
        self.rhs = None
        self.u = None
        self.ud = None
        self.udd = None
        self.displacements = np.empty((self.total_steps, self.model.total_DOFs))
        self.velocities = np.empty((self.total_steps, self.model.total_DOFs))
        self.accelerations = np.empty((self.total_steps, self.model.total_DOFs))
    
    def calculate_coefficients(self):
        alpha = self.alpha
        delta = self.delta
        timestep = self.timestep
        a0 = 1 / (alpha * timestep * timestep)
        a1 = delta / (alpha * timestep)
        a2 = 1 / (alpha * timestep)
        a3 = 1 / (2 * alpha) - 1
        a4 = delta/alpha - 1
        a5 = timestep * 0.5 * (delta/alpha - 2)
        a6 = timestep * (1 - delta)
        a7 = delta * timestep
        self.alphas = (a0, a1, a2, a3, a4, a5, a6, a7)
    
    def build_matrices(self):
        """
        Makes the proper solver-specific initializations before the 
        solution of the linear system of equations. This method MUST be called 
        before the actual solution of the aforementioned system
        """
        provider = self.provider
        a0, a1 = self.alphas[:2]
#        coeffs = {'mass' : a0, 'damping' : a1, 'stiffness' : 1}
#        self.linear_system.Matrix = provider.linear_combination_into_stiffness(coeffs)
        self.linear_system.matrix = (provider._stiffness_matrix 
                                    + a0 * provider._mass_matrix
                                    + a1 * provider._damping_matrix)
   
    def initialize(self, is_first_analysis=True):
        """
        Initializes the models, the solvers, child analyzers, builds
        the matrices, assigns loads and initializes right-hand-side vectors.
        """
        linear_system = self.linear_system
        model = self.model
        if is_first_analysis:
            model.connect_data_structures()

        linear_system.reset()

        
        model.assign_loads()
        
        
        self.initialize_internal_vectors() # call BEFORE build_matrices & initialize_rhs
        self.build_matrices()

        
        self.initialize_rhs()
        
        self.child.initialize()
        
        
        
    def solve(self):
        """
        Solves the linear system of equations by calling the corresponding 
        method of the specific solver attached during construction of the
        current instance.
        """
        # initialize functions to avoid self.function() overhead
        get_rhs_from_history_load = self.provider.get_rhs_from_history_load
        calculate_rhs_implicit = self.calculate_rhs_implicit
        child_solve = self.child.solve
        update_velocity_and_acceleration = self.update_velocity_and_accelaration
        store_results = self.store_results

        for i in range(1, self.total_steps):
            
            self.rhs = get_rhs_from_history_load(i)
            
            self.linear_system.rhs = calculate_rhs_implicit(add_rhs=True)            
            child_solve()
            
            update_velocity_and_acceleration()
            store_results(i)

        
    def calculate_rhs_implicit(self, add_rhs=True):
        """
        Calculates the right-hand-side of the implicit dynamic method. 
        This will be used for the solution of the linear dynamic system.
        """
        alphas = self.alphas
        provider = self.provider
        u = self.u
        ud = self.ud
        udd = self.udd

        udd_eff = alphas[0] * u + alphas[2] * ud + alphas[3] * udd
        ud_eff = alphas[1] * u + alphas[4] * ud + alphas[5] * udd
        
        inertia_forces = provider.mass_matrix_vector_product(udd_eff)
        damping_forces = provider.damping_matrix_vector_product(ud_eff)
        rhs_effective = inertia_forces + damping_forces

        if add_rhs:
             #rhs_effective = uum + ucc
            rhs_effective += self.rhs     

        return rhs_effective
    
    
    def initialize_internal_vectors(self, u0=None, ud0=None):
        if self.linear_system.solution is not None:
            self.linear_system.reset()
            
        total_DOFs = self.model.total_DOFs
        
        if u0 is None:
            u0 = np.zeros((total_DOFs, 1))            
        if ud0 is None:
            ud0 = np.zeros((total_DOFs, 1))
        
        stiffness = self.provider.stiffness_matrix
        mass = self.provider.mass_matrix
        damping = self.provider.damping_matrix #after M and K !
        
        rhs0 = self.provider.get_rhs_from_history_load(0)
        self.linear_system.rhs = rhs0 - stiffness @ u0 - damping @ ud0
        self.linear_system.matrix = mass
        self.solver.initialize()
        self.solver.solve()
        self.udd = self.linear_system.solution
        self.ud = ud0
        self.u = u0
        self.linear_system.reset()


    def initialize_rhs(self):
        self.linear_system.rhs = self.provider.get_rhs_from_history_load(1)


    def update_result_storages(self):
        pass

    def update_velocity_and_accelaration(self):
        a0, a2, a3, a6, a7 = [self.alphas[i] for i in [0,2,3,6,7]]
        
        udd = self.udd
        ud = self.ud 
        u = self.u
        
        u_next = self.linear_system.solution

        udd_next = a0 * (u_next - u) - a2 * ud - a3 * udd 
        ud_next = ud + a6 * udd +  a7 * udd_next
        

        self.u = u_next
        self.ud = ud_next
        self.udd = udd_next
    
    def store_results(self, timestep):
        
        self.displacements[timestep, :] = self.u.ravel()
        self.velocities[timestep, :] = self.ud.ravel()
        self.accelerations[timestep, :] = self.udd.ravel()
        
        

