class Analyzer:
    """Abstract class for Parent Analyzers"""
    
    def __init__(self, provider, child_analyzer):
        """
        Creates an instance that uses a specific problem type and an
        appropriate child analyzer for the construction of the system of 
        equations arising from the actual physical problem.
        
        Parameters
        ----------
        provider : ProblemStructural
            Instance of the problem type to be solved.
        child_analyzer : Analyzer
            Instance of the child analyzer that will handle the solution of
            the system of equations.
        linear_system 
            Instance of the linear system that will be initialized.
        """
        self.provider = provider    
        self.child = child_analyzer
        self.child.parent = self       

 
    def build_matrices(self):
        """
        Builds the appropriate linear system matrix and updates the 
        linear system instance used in the constructor.
        """
        return self.provider.calculate_matrix(self.linear_system)



    

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
        provider : ProblemStructural
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
        return self.provider.calculate_matrix(self.linear_system)

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
    
    def __init__(self, model, solver, provider, child_analyzer, timestep, total_time, alpha, delta):
        
        super().__init__(provider, child_analyzer)
        
        self.model = model
        self.solver = solver
        try:
            self.linear_systems = solver.linearsystems
        except:
            pass
        self.timestep = timestep
        self.total_time = total_time
        self.alpha = alpha
        self.delta = delta
    
    def build_matrices(self):
        pass