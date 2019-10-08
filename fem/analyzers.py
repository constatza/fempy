

from ABC import abc, abstractmethod
import numpy as np


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


class DynamicNewmark(Analyzer):
    """Implements the Newmark method for dynamical analysis."""
    
    def __init__(self, model, solver , provider, child_analyzer, timestep, total_time, alpha, delta):
        
        super().__init__(provider, child_analyzer)
        
        self.timestep = timestep
        self.total_time = total_time
        self.alpha = alpha
        self.delta = delta
        self.calculate_coefficients()
        self.linear_system = solver.linear_system
    
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
        
        a0, a1 = self.alphas[:2]
#        coeffs = {'mass' : a0, 'damping' : a1, 'stiffness' : 1}
#        self.linear_system.Matrix = provider.linear_combination_into_stiffness(coeffs)
        self.linear_system.Matrix = provider.stiffness_matrix 
                                    + a0 * provider.mass_matrix
                                    + a1 * provider.damping_matrix
        
    def get_other_rhs_components(self, linear_system, current_solution):
        """Calculates inertia forces and damping forces."""
       alphas = self.alphas
       
       u  = current_solution[0]
       v = currnet_solution[1]
       a = current_solution[2]
       inertia_forces = provider.mass_matrix @ ( alphas[0]*u + alphas[2]*v + alphas[3]*a)
       damping_forces = provider.damping_matrix  @ ( alphas[1]*u + alphas[4]*v + alphas[5]*a)
       return interia_forces + damping_forces   
   
    def initialize(self, is_first_analysis=True):
        """
        Initializes the models, the solvers, child analyzers, builds
        the matrices, assigns loads and initializes right-hand-side vectors.
        """
        if is_first_analysis:
            model.connect_data_structures()
            
        linear_system.reset()
        linear_system.forces = np.zeros(linear_system.size)  
        self.build_matrices()
        model.assign_loads(solver.DistributeNodalLoads) # ?????
        linear_system.rhs = model.forces
        self.initialize_internal_vectors()
        initialize_rhs()
        child.initialize(is_first_analysis)
            
        
    def solve(self):
        """
        Solves the linear system of equations by calling the corresponding 
        method of the specific solver attached during construction of the
        current instance.
        """
        num_timsteps = int(total_time / self.timestep)
        for i in range(num_timesteps):
        
            print("Newmark step: {0}", i)
            
            
            rhs = self.provider.get_rhs_from_history_load(i)
            linear_system.rhs = rhs
            self.initialize_rhs()
            self.calculcate_rhs_implicit(rhs, v, v1, v2, self.mass_matrix)            
            self.child_analyzer.Solve()
            self.update_velocity_and_accelaration(i)
    
    def calculate_rhs_implicit(rhs, v, v1, v2, mass_matrix, dump_matrix):
        """
        Calculates the right-hand-side of the implicit dyanmic method. 
        This will be used for the solution of the linear dynamic system.
        """
        uu = a0 * v + a2 * v1 + a3 * v2
        uc = a1 * v + a4 * v1 + a5 * v2
        
        uum = mass_matrix @ uu
        ucc = dump_matrix @ uc

       
        rhs_effective = rhs + uum + ucc
      
        #rhs_effective = uum + ucc
       
        return rhs_effective
    
    
    def build_matrices(self):
        """
        Builds the appropriate linear system matrix and updates the 
        linear system instance used in the constructor.
        """
        return self.provider.calculate_matrix(self.linear_system)
    