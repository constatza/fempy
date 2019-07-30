

import numpy as np


class Solver:
    
    def __init__(self, square_matrix, target_vector):
        self.matrix = square_matrix
        self.vector = target_vector




class SimpleSolver(Solver):
    
    def __init__(self, A,b):
        Solver.__init__(square_matrix=A, target_vector=b)

    def solve(self):
        return np.linalg.solve(self.matrix, self.vector)