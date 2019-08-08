

import scipy as sp

class Solver:
    
    def __init__(self, system):
        self.system = system
    
    def initialize(self):
        """makes preparations."""
        pass


class SimpleSolver(Solver):
    
    def __init__(self, linear_system):
        Solver.__init__(self, linear_system)

    def solve(self):
        self.system.solution = sp.linalg.solve(self.system.matrix, 
                                          self.system.rhs,
                                          assume_a='sym')


class LinearSystem:
    """
    This class represents a linear system as a collection
    of a matrix and a right-hand side vector
    """
    
    def __init__(self, rhs):
        """
        Initializes a linear system with a specific right-hand side
        
        Parameters
        ----------
        rhs : np.ndarray
            Vector representing the right-hand side of the linear system
        solution : np.ndarray
            Vector representing the solution of the linear system
        matrix2D : np.ndarray
            2D matrix of the linear system
        """
        self.rhs = rhs
        self.solution = None #np.empty(rhs.shape)
        self.matrix = None
    

