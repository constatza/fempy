from dataclasses import dataclass

class LinearSystem:
    """
    This class represents a linear system as a collection
    of a matrix and a right-hand side vector
    """
    
    def __init__(self, rhs):
        """
        Initializes a linear system with a specific right-hand side.
        
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
        self.solution = None 
        self.matrix = None
    