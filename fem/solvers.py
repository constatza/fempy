

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

    def solve(self):
#        S = sparse.bsr_matrix(self.linear_system.matrix)
        L = linalg.cho_factor(self.linear_system.matrix)
        solution = linalg.cho_solve(L, self.linear_system.rhs)
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
    
    def solve(self):
        sparseM = sparse.csc_matrix(self.linear_system.matrix)
        solveLU = splinalg.factorized(sparseM)
        self.linear_system.solution = solveLU(self.linear_system.rhs)
     


