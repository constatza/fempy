

import scipy as sp

class Solver:
    
    def __init__(self, square_matrix, right_hand_side):
        self.matrix = square_matrix
        self.vector = right_hand_side


class SimpleSolver(Solver):
    
        Solver.__init__(square_matrix=linear_system.matrix2D, 
                        right_hand_side=linear_system.rhs)

    def solve(self):
        return sp.linalg.solve(self.matrix, self.vector, assume_a='sym')


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
        self.matrix2D = None
    

