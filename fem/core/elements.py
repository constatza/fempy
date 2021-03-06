import numpy as np
import numba as nb
import numba.cuda as cuda

from fem.core.gauss import Quadrature, Point3D
from fem.assemblers import GenericDOFEnumerator
from fem.core.entities import Element, DOFtype



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
#        current_node_coordinates = self.get_current_node_coordinates()
#        coordinates_changed  = (self.node_coordinates!=current_node_coordinates).any()
        if (self._integration_points==None):#| coordinates_changed       
#            self._node_coordinates = current_node_coordinates
            self._integration_points = self.calculate_gauss_matrices(self.node_coordinates)            
        return self._integration_points
        
    @staticmethod
    def calculate_shape_functions(ksi, eta):
        """
        Calculates the shape functions' values
        at an integration point.

       N = [N1, N2 N3, N4] = 1/4 * [(1-ksi)(1-eta),
                                    (1+ksi)(1-eta),
                                    (1+ksi)(1+eta),
                                    (1-ksi)(1+eta) ]
       Returns:
       --------
       N : ndarray
           Array with size 1x4 containing the shape functions.
        
        """
        
        fN05 = 0.5
        f_ksi_plus = (1.0 + ksi) * fN05
        f_eta_plus = (1.0 + eta) * fN05
        f_ksi_minus = (1.0 - ksi) * fN05
        f_eta_minus = (1.0 - eta) * fN05
        
        N = np.empty((4,), dtype=np.float64)
        N[0] = f_ksi_minus * f_eta_minus
        N[1] = f_ksi_plus * f_eta_minus
        N[2] = f_ksi_plus * f_eta_plus
        N[3] = f_ksi_minus * f_eta_plus
        
        return N   
    
    @staticmethod
    def calculate_shape_function_derivatives(ksi, eta):
        """
        Calculates the shape function derivatives' values
        at an integration point.
        
        Dn = .25*[ N1,ksi  N2,ksi  N3,ksi  N4,ksi ]
                 [ N1,eta  N2,eta  N3,eta  N4,eta ]
        
           = .25*[-(1-eta), (1-eta), (1+eta), -(1+eta)]
                 [-(1-ksi), -(1+ksi), (1+ksi), (1-ksi)]
        
           = [-f_eta_minus, f_eta_minus, f_eta_plus, -f_eta_plus]
             [-f_ksi_minus, -f_ksi_plus, f_ksi_plus, f_ksi_minus]
             
        Returns:
        --------
        Dn : ndarray
           Array with size 2x4 containing the shape functions' derivatives.
        
        """
        
        fN025 = 0.25
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
        integration_points_per_axis = Quadrature.get_gauss_points(2)
        #total_sampling_points = len(integration_points_per_axis)**2
        integration_points_list = []
        for point_ksi in integration_points_per_axis:
            for point_eta in integration_points_per_axis:
                
                ksi = point_ksi.coordinate
                eta = point_eta.coordinate
                
                shape_funcs = Quad4.calculate_shape_functions(ksi, eta)
                shape_func_der = Quad4.calculate_shape_function_derivatives(ksi, eta)
                
                J = Quad4.calculate_jacobian_matrix(shape_func_der, node_coordinates)
                detJ = Quad4.calculate_determinant(J)
                deform_matrix = Quad4.calculate_deformation_matrix(ksi, eta, J, 1/detJ, shape_func_der)
                
                weight_factor = point_ksi.weight * point_eta.weight * detJ
                current_gauss_point = Point3D(ksi, eta, 0, shape_funcs, deform_matrix, weight_factor)
                integration_points_list.append(current_gauss_point)
                
        return integration_points_list
    
    @staticmethod
    def calculate_stiffness_matrix(integration_points, materials_at_gauss_points, thickness):
        """Method that calculates the stiffness matrix of an isoparametric 
        4-noded quadrilateral element, with constant thickness.
        """

        
        Es = np.empty((3 ,3, Quad4.gauss_iter2))
        Bs = np.empty((3, 8, Quad4.gauss_iter2))
        ws = np.empty(Quad4.gauss_iter2)
        pointID = -1
        for point in integration_points:
            pointID += 1
            constitutive_matrix = materials_at_gauss_points[pointID].constitutive_matrix
            deformation_matrix = point.deformation_matrix
            ws[pointID] = point.weight
            Es[:, :, pointID] = constitutive_matrix
            Bs[:, :, pointID] = deformation_matrix
            
        stiffness_matrix = Quad4.sum_stiffnesses(Es, Bs, ws, thickness)
        return stiffness_matrix
    
    @staticmethod
    @nb.njit('float64[:,:](float64[:,:,:], float64[:,:,:], float64[:], float64)')
    def sum_stiffnesses(Es, Bs, ws, thickness):
        stiffness_matrix = np.zeros((8,8))
        EB = np.empty(3)
        for k in range(ws.shape[0]):
            for i in range(Bs.shape[1]):
                
                
                EB[0] = (Es[0, 0, k] * Bs[0, i, k]
                        + Es[0, 1, k] * Bs[1, i, k]
                        + Es[0, 2, k] * Bs[2, i, k])
                
                EB[1] = (Es[1, 0, k] * Bs[0, i, k]
                        + Es[1, 1, k] * Bs[1, i, k]
                        + Es[1, 2, k] * Bs[2, i, k])
                
                EB[2] = (Es[2, 0, k] * Bs[0, i, k]
                        + Es[2, 1, k] * Bs[1, i, k]
                        + Es[2, 2, k] * Bs[2, i, k])
                    
                for j in range(i, Bs.shape[1]):
                    stiffness = (Bs[0, j, k] * EB[0] 
                                + Bs[1, j, k] * EB[1]
                                + Bs[2, j, k] * EB[2])* ws[k] * thickness
                    
                    stiffness_matrix[i, j] += stiffness 
                    stiffness_matrix[j, i] = stiffness_matrix[i, j]
        
        return stiffness_matrix

    @staticmethod
    def calculate_mass_matrix(integration_points, mass_density, thickness):
        """Method that calculates the mass matrix of an isoparametric 
        4-noded quadrilateral element, with constant thickness.
        """

        Ns = np.empty((Quad4.gauss_iter2, 4))
        ws = np.empty(Quad4.gauss_iter2)
        pointID = -1
        for point in integration_points:
            pointID += 1
            ws[pointID] = point.weight
            Ns[pointID, :] = point.shape_functions
            
        mass_matrix = Quad4.sum_masses(Ns, ws, mass_density, thickness)
        
        return mass_matrix

    @staticmethod
    @nb.njit('float64[:,:](float64[:,:], float64[:], float64, float64)')
    def sum_masses(Ns, ws, mass_density, thickness):
        
        mass_matrix = np.zeros((8,8))
        
        w1 = mass_density * thickness
        for i in range(ws.shape[0]):
            
            w2 = ws[i] * w1
            
            NN_1j = Ns[i, 0] * Ns[i, :] * w2
            NN_2j = Ns[i, 1] * Ns[i, :] * w2
            NN_3j = Ns[i, 2] * Ns[i, :] * w2
            NN_4j = Ns[i, 3] * Ns[i, :] * w2
            
            mass_matrix[0, 0:7:2] += NN_1j
            mass_matrix[1, 1:8:2] += NN_1j
            mass_matrix[2, 0:7:2] += NN_2j
            mass_matrix[3, 1:8:2] += NN_2j
            mass_matrix[4, 0:7:2] += NN_3j
            mass_matrix[5, 1:8:2] += NN_3j
            mass_matrix[6, 0:7:2] += NN_4j
            mass_matrix[7, 1:8:2] += NN_4j
        
        return mass_matrix

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
    
    @staticmethod
    def mass_matrix(element):
        mass_matrix = Quad4.calculate_mass_matrix(element.integration_points,
                                                  element.material.mass_density,
                                                  element.thickness)
        return mass_matrix

