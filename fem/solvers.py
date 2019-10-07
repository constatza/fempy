

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
     


