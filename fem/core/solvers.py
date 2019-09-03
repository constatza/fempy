

from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg


class Solver:
    
    def __init__(self, system):
        self.system = system
    
    def initialize(self):
        """makes preparations."""
        pass

class ConjugateGradientSolver(Solver):
    
    def __init__(self, linear_system):
        super().__init__(linear_system)

    
    def solve(self):
        self.solution = splinalg.cg(self.system.matrix, self.system.rhs)

class CholeskySolver(Solver):
    
    def __init__(self, linear_system):
        super().__init__(linear_system)

    def solve(self):
#        S = sparse.bsr_matrix(self.system.matrix)
        L = linalg.cho_factor(self.system.matrix)
        self.system.solution = linalg.cho_solve(L, self.system.rhs).ravel()

class SparseSolver(Solver):
    def __init__(self, linear_system):
        super().__init__(linear_system)
    
    def solve(self):
        sparseM = sparse.csr_matrix(self.system.matrix)
        self.system.solution = splinalg.spsolve(sparseM, self.system.rhs)

class SparseLUSolver(Solver):
    def __init__(self, linear_system):
        super().__init__(linear_system)
    
    def solve(self):
        sparseM = sparse.csc_matrix(self.system.matrix)
        solveLU = splinalg.factorized(sparseM)
        self.system.solution = solveLU(self.system.rhs).ravel()
     
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
    

