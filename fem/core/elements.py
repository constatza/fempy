
import numpy as np
import numba as nb
from .entities import Element, DOFtype, GaussQuadrature, GaussPoint3D
from .assemblers import GenericDOFEnumerator

class Quad4(Element):
    """Quadrilateral 4-node 2D element"""
    
    nodal_DOFtypes = [DOFtype.X, DOFtype.Y]
    
    DOFtypes = [nodal_DOFtypes,
                nodal_DOFtypes,
                nodal_DOFtypes,
                nodal_DOFtypes]
    
    gauss_iter = 2
    gauss_iter2 = 4
    gauss_iter3 = 8
    
    def __init__(self, *args, material=None ,thickness=None, DOF_enumerator=GenericDOFEnumerator(), **kwargs):
        
        super().__init__(*args, **kwargs)
        self.material = material   
        self.thickness = thickness
        self.DOF_enumerator = DOF_enumerator
        self._node_coordinates = None
        self._integration_points = None
        self._stiffness_matrix = None
    
    @property
    def materials_at_gauss_points(self):
        """A 4-list containing 4 copies of the material"""
        return self._materials_at_gauss_points
    
    @materials_at_gauss_points.setter
    def materials_at_gauss_points(self, material):
        materials = []
        for i in range(Quad4.gauss_iter2):
            materials.append(material)
        self._materials_at_gauss_points = materials
    
    @property
    def material(self):
        return self._material
    
    @material.setter
    def material(self, material):
        self._material = material
        self.materials_at_gauss_points = material
        
    @property
    def node_coordinates(self):
        if self._node_coordinates is None:
            self._node_coordinates = self.get_current_node_coordinates()
        return self._node_coordinates
            
    def get_current_node_coordinates(self):
        XY = np.empty((4,2))
        for i, node in enumerate(self.nodes):
            XY[i, 0] = node.X
            XY[i, 1] = node.Y
        return XY 
    
    @property
    def integration_points(self):
        current_node_coordinates = self.get_current_node_coordinates()
        coordinates_changed  = (self.node_coordinates==current_node_coordinates).all()
        if coordinates_changed | (self._integration_points==None):
            self._node_coordinates = current_node_coordinates
            self._integration_points = self.calculate_gauss_matrices(current_node_coordinates)            
        return self._integration_points
        
        
    @staticmethod
    def calculate_shape_function_derivatives(ksi, eta):
        """
        Calculates the shape function derivative values
        for an integration point.
        """
        fN025 = 0.25;
        f_ksi_plus = (1.0 + ksi) * fN025
        f_eta_plus = (1.0 + eta) * fN025
        f_ksi_minus = (1.0 - ksi) * fN025
        f_eta_minus = (1.0 - eta) * fN025
        
        Dn = np.empty((2,4), dtype=np.float64)
        Dn[0, 0] = -f_eta_minus
        Dn[0, 1] = f_eta_minus
        Dn[0, 2] = f_eta_plus
        Dn[0, 3] = -f_eta_plus
        
        Dn[1, 0] = -f_ksi_minus
        Dn[1, 1] = -f_ksi_plus
        Dn[1, 2] = f_ksi_plus
        Dn[1, 3] = f_ksi_minus

#        Dn =  np.array([[-f_eta_minus, f_eta_minus, f_eta_plus, -f_eta_plus], 
#                        [-f_ksi_minus, -f_ksi_plus, f_ksi_plus, f_ksi_minus]])
        
        
#       Dn = .25*[-(1-eta), (1-eta), (1+eta), -(1+eta),
#                 -(1-ksi), -(1+ksi), (1+ksi), (1-ksi)]
#
#       Dn = .25*[N1,ksi N2,ksi N3,ksi N4,ksi]
#                [N1,eta N2,eta N3,eta N4,eta]  
        return Dn
    
    @staticmethod
    def calculate_deformation_matrix(ksi, eta, jacobian_matrix, detJ_inv, shape_function_derivatives_matrix):
        """ deformation matrix B = B1 @ B2 
        
        
        B1 = 1/detJ *[[J[1, 1], -J[0, 1],       0,         0],
                      [       0,       0, -J[1, 0],   J[0, 0]],
                      [-J[1, 0],  J[0, 0],  J[1, 1], -J[0, 1]]]
        
        
        B2 =.25*[[-1+eta, 0, 1-eta, 0, 1+eta, 0, -1-eta, 0],
                 [-1+ksi, 0, -1-ksi, 0, 1+ksi, 0, 1-ksi, 0],
                 [0, -1+eta, 0, 1-eta, 0, 1+eta, 0, -1-eta],
                 [0, -1+ksi, 0, -1-ksi, 0, 1+ksi, 0, 1-ksi]]
        """
        B1 = np.zeros((3, 4), dtype=np.float64)
        
        B_matrix = np.zeros((3, 8), dtype=np.float64)

        
        B1[0, 0] = detJ_inv * jacobian_matrix[1, 1]
        B1[0, 1] = -detJ_inv * jacobian_matrix[0, 1]
        B1[1, 2] = -detJ_inv * jacobian_matrix[1, 0]
        B1[1, 3] = detJ_inv * jacobian_matrix[0, 0]
        B1[2, 0] = -detJ_inv * jacobian_matrix[1, 0]
        B1[2, 1] = detJ_inv * jacobian_matrix[0, 0]
        B1[2, 2] = detJ_inv * jacobian_matrix[1, 1]
        B1[2, 3] = -detJ_inv * jacobian_matrix[0, 1]
    
        dN1 = shape_function_derivatives_matrix[:, 0]
        dN2 = shape_function_derivatives_matrix[:, 1]
        dN3 = shape_function_derivatives_matrix[:, 2]
        dN4 = shape_function_derivatives_matrix[:, 3]
        
        B_matrix[0, 0] = B1[0, 0] * dN1[0] + B1[0, 1] * dN1[1]
        B_matrix[0, 2] = B1[0, 0] * dN2[0] + B1[0, 1] * dN2[1]
        B_matrix[0, 4] = B1[0, 0] * dN3[0] + B1[0, 1] * dN3[1]
        B_matrix[0, 6] = B1[0, 0] * dN4[0] + B1[0, 1] * dN4[1]
        
        B_matrix[1, 1] = B1[1, 2] * dN1[0] + B1[1, 3] * dN1[1]
        B_matrix[1, 3] = B1[1, 2] * dN2[0] + B1[1, 3] * dN2[1]
        B_matrix[1, 5] = B1[1, 2] * dN3[0] + B1[1, 3] * dN3[1]
        B_matrix[1, 7] = B1[1, 2] * dN4[0] + B1[1, 3] * dN4[1]
        
        B_matrix[2, 0] = B1[2, 0] * dN1[0] + B1[2, 1] * dN1[1]
        B_matrix[2, 2] = B1[2, 0] * dN2[0] + B1[2, 1] * dN2[1]
        B_matrix[2, 4] = B1[2, 0] * dN3[0] + B1[2, 1] * dN3[1]
        B_matrix[2, 6] = B1[2, 0] * dN4[0] + B1[2, 1] * dN4[1]
        
        B_matrix[2, 1] = B1[2, 2] * dN1[0] + B1[2, 3] * dN1[1]
        B_matrix[2, 3] = B1[2, 2] * dN2[0] + B1[2, 3] * dN2[1]
        B_matrix[2, 5] = B1[2, 2] * dN3[0] + B1[2, 3] * dN3[1]
        B_matrix[2, 7] = B1[2, 2] * dN4[0] + B1[2, 3] * dN4[1]
        
        return B_matrix

    @staticmethod
    def calculate_jacobian_matrix(shape_function_derivatives, node_coordinates):
        """Calculates the jacobian matrix for an integration point."""
        J = shape_function_derivatives @ node_coordinates
        return J

    @staticmethod
    def calculate_determinant(jacobian):
        """Calculates the Jacobian determinant for an integration point."""
        detJ = jacobian[0,0]*jacobian[1,1] - jacobian[1,0]*jacobian[0,1]
        if detJ<1e-7:
            raise ValueError("Jacobian negative or close to zero!")
        return detJ

    @staticmethod
    def calculate_jacobian_inv(jacobian_matrix, jacobian_determinant):
        """Calculates the inverse jacobian matrix.
        
           [J]^-1 = 1/det[J] [ J22    -J12]
                             [-J21     J11]           
        """
        J_inv = np.empty((2,2), dtype=np.float64)
        detJ_inv = 1.0 / jacobian_determinant
        J_inv[0, 0] = jacobian_matrix[1, 1] * detJ_inv
        J_inv[0, 1] = -jacobian_matrix[0, 1]* detJ_inv
        J_inv[1, 0] = -jacobian_matrix[1, 0] * detJ_inv
        J_inv[1, 1] = jacobian_matrix[0, 0] * detJ_inv
        return J_inv
    
    def calculate_gauss_matrices(self, node_coordinates):
        """ Calculates the integration points. """
        integration_points_per_axis = GaussQuadrature.get_gauss_points(2)
        #total_sampling_points = len(integration_points_per_axis)**2
        integration_points_list = []
        for point_ksi in integration_points_per_axis:
            for point_eta in integration_points_per_axis:
                ksi = point_ksi.coordinate
                eta = point_eta.coordinate
                SF_der = Quad4.calculate_shape_function_derivatives(ksi, eta)
                J = Quad4.calculate_jacobian_matrix(SF_der, node_coordinates)
                detJ = Quad4.calculate_determinant(J)
                DM = Quad4.calculate_deformation_matrix(ksi, eta,
                                                       J, 1/detJ, SF_der)
                weight_factor = point_ksi.weight * point_eta.weight * detJ
                current_gauss_point =  GaussPoint3D(ksi, eta, 0, DM, weight_factor)
                integration_points_list.append(current_gauss_point)
        return integration_points_list
    
    @staticmethod
    #@nb.njit
    def calculate_stiffness_matrix(integration_points, materials_at_gauss_points, thickness):
        """Method that calculates the stiffness matrix of an isoparametric 
        4-noded quadrilateral element, with constant thickness.
        """
        #integration_points = integration_points
        stiffness_matrix = np.zeros((8,8))
        point_ID = -1
        for point in integration_points:
            point_ID += 1
            constitutive_matrix = materials_at_gauss_points[point_ID].constitutive_matrix
            deformation_matrix = point.deformation_matrix
            stiffness_matrix += deformation_matrix.T @ constitutive_matrix @ deformation_matrix * point.weight
        stiffness_matrix *= thickness    
        return stiffness_matrix
    
    
    def get_stiffness_matrix(self):
        stiffness_matrix = self.calculate_stiffness_matrix(self.integration_points,   
                                                       self.materials_at_gauss_points,
                                                       self.thickness)
        return stiffness_matrix
    
    @staticmethod
    def stiffness_matrix(element):
        stiffness_matrix = Quad4.calculate_stiffness_matrix(element.integration_points,
                                                            element.materials_at_gauss_points,
                                                            element.thickness)
        return stiffness_matrix
        