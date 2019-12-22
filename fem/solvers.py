
import numpy as np
from scipy import linalg 
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse import linalg as splinalg
# from sksparse.cholmod import cholesky


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




# class SparseCholeskySolver(Solver):
    
#     def __init__(self, linear_system):
#         super().__init__(linear_system)
    
#     def initialize(self):
#         """ Factorizes linear system's matrix once for many different rhs."""
#         sparse_matrix = csc_matrix(self.linear_system.matrix)
#         factor = cholesky(sparse_matrix)
#         self.sparse_cho_solve = factor
        
    
#     def solve(self):
#         self.linear_system.solution = self.sparse_cho_solve(self.linear_system.rhs)
     

class SparseLUSolver(Solver):
    
    def __init__(self, linear_system):
        super().__init__(linear_system)
    
    def initialize(self):
        """ Factorizes linear system's matrix once for many different rhs."""
        sparse_matrix = csc_matrix(self.linear_system.matrix)
        lu = splinalg.splu(sparse_matrix)
        self.sparse_lu = lu
        
    
    def solve(self):
        self.linear_system.solution = self.sparse_lu.solve(self.linear_system.rhs)
    
