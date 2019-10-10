
""" Gauss integration library"""

class Point1D:
    """Defines an one-dimensional Gauss Legendre integration point."""
    
    def __init__(self, coordinate=None, weight=None):
        self.coordinate = coordinate
        self.weight = weight

       
class Point3D:
    """Defines a three-dimensional Gauss Legendre integration point."""
    
    def __init__(self, ksi, eta, zeta, shape_functions, deformation_matrix, weight):	
        self.ksi = ksi
        self.eta = eta
        self.zeta = zeta
        self.shape_functions = shape_functions
        self.deformation_matrix = deformation_matrix
        self.weight = weight
        

class Quadrature:
    """Provides one-dimensional Gauss-Legendre points and weights."""

    gauss_point1 = Point1D(coordinate=0, weight=2)
    
    gauss_point2a = Point1D(coordinate=-.5773502691896, weight=1)    
    gauss_point2b = Point1D(coordinate=0.5773502691896, weight=1)

    
    @staticmethod
    def get_gauss_points(integration_degree):
        """
         For point coordinates, we encounter the following constants:
         0.5773502691896 = 1 / Square Root 3
         0.7745966692415 = (Square Root 15)/ 5
         0.8611363115941 = Square Root( (3 + 2*sqrt(6/5))/7)
         0.3399810435849 = Square Root( (3 - 2*sqrt(6/5))/7)
         
         For the weights, we encounter the followings constants:
         0.5555555555556 = 5/9
         0.8888888888889 = 8/9
         0.3478548451375 = (18 - sqrt30)/36
         0.6521451548625 = (18 + sqrt30)/36  
        """
        if integration_degree==1:
            return [Quadrature.gauss_point1]
        elif integration_degree==2:
            return [Quadrature.gauss_point2a,
                    Quadrature.gauss_point2b]
        else:
            raise NotImplementedError("Unsupported degree of integration: {:}".format(integration_degree))
            