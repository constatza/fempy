

from scipy import linalg, sparse
from scipy.sparse import linalg as splinalg
import numpy as np

class Solver:
    
    def __init__(self, linear_system):
        self.linear_system = linear_system
    
    def initialize(self):
        """makes preparations."""
        pass

class ConjugateGradientSolver(Solver):
    
    def __init__(self, linear_system):
        super().__init__(linear_system)

    
    def solve(self):
        solution = splinalg.cg(self.linear_system.matrix, self.linear_system.rhs)
        self.linear_system.solution = solution

class CholeskySolver(Solver):
    
    def __init__(self, linear_system):
        super().__init__(linear_system)
        
    def initialize(self):
        """ Factorizes linear system's matrix once for many different rhs."""
        self.L = linalg.cho_factor(self.linear_system.matrix, check_finite=False)
    
    def solve(self):
        solution = linalg.cho_solve(self.L, self.linear_system.rhs, check_finite=False)
        self.linear_system.solution = solution


class SparseSolver(Solver):
    def __init__(self, linear_system):
        super().__init__(linear_system)
    
    def solve(self):
        sparseM = sparse.csr_matrix(self.linear_system.matrix)
        self.linear_system.solution = splinalg.spsolve(sparseM, self.linear_system.rhs)

class SparseLUSolver(Solver):
    
    def __init__(self, linear_system):
        super().__init__(linear_system)
    
    def initialize(self):
        """ Factorizes linear system's matrix once for many different rhs."""
        sparseM = sparse.csr_matrix(self.linear_system.matrix)
        self.sparse_solveLU = splinalg.factorized(sparseM)
        
    
    def solve(self):
       
        self.linear_system.solution = self.sparse_solveLU(self.linear_system.rhs)
     

