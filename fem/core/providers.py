# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:49:31 2019

@author: constatza
"""

from fem.assemblers import GlobalMatrixAssembler
from numpy import array, zeros, sqrt, eye, abs, max
import scipy.linalg as linalg
import scipy.sparse.linalg as splinalg
from scipy.sparse import csc_matrix


class ElementStiffnessProvider:
    """ Responsible for providing elemental stiffness matrix 
    for the global matrix assembly.
    """
    @staticmethod
    def matrix(element):
        return element.element_type.stiffness_matrix(element)


class ElementMaterialOnlyStiffnessProvider:
    """ Responsible for providing elemental stiffness matrix 
    for the global matrix assembly.
        
        Deploys the fact that only the material has changed.
    """
    @staticmethod
    def matrix(element):
        points = element.integration_points
        thickness = element.thickness
        element.material.constitutive_matrix = None
        materials = element.materials_at_gauss_points
        return element.calculate_stiffness_matrix(points, materials, thickness)


class ElementMassProvider:
    """ Responsible for providing elemental mass matrix 
    for the global matrix assembly.
        
        Deploys the fact that only the material has changed.
    """
    @staticmethod
    def matrix(element):
        return element.element_type.mass_matrix(element)



class GlobalMatrixProvider:
    
    def __init__(self):
        pass
    
    @staticmethod
    def build_rhs_from_history_loads(timestep, static_forces, dynamic_forces,
                                   inertia_loads, inertia_direction_vectors, mass_matrix):
        
        nodal_rhs = GlobalMatrixProvider.build_rhs_from_nodal_loads
        inertia_rhs = GlobalMatrixProvider.build_rhs_from_inertia_loads
        
        nodal = nodal_rhs(timestep,
                          static_forces,
                          dynamic_forces)
        
        inertia = inertia_rhs(timestep,
                              inertia_loads,
                              inertia_direction_vectors,
                              mass_matrix)
        return nodal + inertia
    
    @staticmethod
    def build_rhs_from_inertia_loads(timestep, inertia_loads,
                                   inertia_direction_vectors, mass_matrix):
        nloads = len(inertia_loads)
        inertia_forces = zeros(mass_matrix.shape[0])
        for i in range(nloads):
            inertia_loads[i].time_history(timestep,
                                 inertia_direction_vectors[:, i],
                                 inertia_forces)
            
            
        return inertia_forces[:, None]
    
    @staticmethod
    def build_rhs_from_nodal_loads(timestep, static_forces, dynamic_forces):
        dynamic_forces_vector = zeros(static_forces.shape, order='F')
        for dof, history in dynamic_forces.items():
            dynamic_forces_vector[dof] = history(timestep)
         
        
        return static_forces + dynamic_forces_vector
    
    @staticmethod
    def build_inertia_direction(model):
        return model.inertia_forces_direction_vectors
    
    @staticmethod
    def build_mass_matrix(model, element_mass_provider):
        mass = GlobalMatrixAssembler.calculate_global_matrix(model, element_mass_provider)
        return mass
     
    @staticmethod   
    def build_stiffness_matrix(model, element_stiffness_provider):
        stiff = GlobalMatrixAssembler.calculate_global_matrix(model, element_stiffness_provider)
        return stiff
    
    @staticmethod
    def build_damping_matrix(damping_provider):
        if damping_provider is None:
            damp = zeros(damping_provider.stiffness_matrix.shape)
        else:
            damp = damping_provider.calculate_global_matrix()
        return damp
    
    @staticmethod
    def get_mass_matrix(mass_matrix):
        return mass_matrix
    
    @staticmethod
    def get_stiffness_matrix(stiffness_matrix):
        return stiffness_matrix
    
    @staticmethod
    def get_damping_matrix(damping_matrix):
        return damping_matrix
    
    @staticmethod 
    def get_rhs_from_history_loads(timestep, problem):
        model = problem.model
        stforces = model.forces
        dyforces = model.dynamic_forces
        inloads = model.inertia_loads
        in_dir_vectors = problem.inertia_vectors  
        rhs = GlobalMatrixProvider.build_rhs_from_history_loads(timestep, 
                                                                stforces, 
                                                                dyforces,
                                                                inloads,
                                                                in_dir_vectors,
                                                                problem._mass_matrix)
        
        return rhs
        

class ReducedGlobalMatrixProvider(GlobalMatrixProvider):
    
    def __init__(self, linear_map):
        self.linear_map = linear_map
        # self.get_mass_matrix = self.reduced_matrix(super().get_mass_matrix)
        # self.get_stiffness_matrix = self.reduced_matrix(super().get_stiffness_matrix)
        # self.get_damping_matrix = self.reduced_matrix(super().get_damping_matrix)
        #self.get_rhs_from_history_loads = self.reduced_vector(super().get_rhs_from_history_loads)
        
    # def reduced_matrix(self, func):
    #     def wrapper(*args, **kwargs):
    #         matrix = func(*args, **kwargs)
    #         return self.linear_map.direct_transform_matrix(matrix)
    #     return wrapper
    
    def reduced_vector(self, func):
        def wrapper(*args, **kwargs):
            vector = func(*args, **kwargs)
            return self.linear_map.transpose_transform_vector(vector)
        return wrapper
    
    
    def get_mass_matrix(self, mass_matrix):
        return self.linear_map.direct_transform_matrix(mass_matrix)
    
    def get_stiffness_matrix(self, stiffness_matrix):
        return self.linear_map.direct_transform_matrix(stiffness_matrix)
    
    def get_damping_matrix(self, damping_matrix):
        return self.linear_map.direct_transform_matrix(damping_matrix)

    def get_rhs_from_history_loads(self, timestep, problem):
        rhs = GlobalMatrixProvider.get_rhs_from_history_loads(timestep, problem)
        rhs_r = self.linear_map.transpose_transform_vector(rhs)
        return rhs_r

class RayleighDampingMatrixProvider:
    
    def __init__(self, coeffs=None):
        self.coeffs = coeffs
        self.stiffness_matrix = None
        self.mass_matrix = None
    
    def calculate_global_matrix(self, stiffness_matrix=None, mass_matrix=None):
        if stiffness_matrix is None:
            stiffness_matrix = self.stiffness_matrix
        if mass_matrix is None:
            mass_matrix = self.mass_matrix
        damping_coeffs = self.coeffs
        k = mass_matrix.shape[0]
        eigvals = linalg.eigvalsh(mass_matrix, 
                                  b=stiffness_matrix,
                                  eigvals=None,#(k-2,k-1),
                                  type=1,
                                  turbo=True, 
                                  check_finite=False,
                                  overwrite_a=True,
                                  overwrite_b=True)

        
        wmegas =1/sqrt(eigvals[-2:])
        self.frequencies = wmegas
        Matrix = .5* array([1/wmegas,wmegas]) 
        a = linalg.solve(Matrix.T, damping_coeffs, check_finite=False)
        
        return a[0]*mass_matrix + a[1]*stiffness_matrix
    
# import scipy
def make_sparse(M):
    m = abs(M)
    maks = max(m)
    is_zero = (m/maks) < 1e-3
    M[is_zero] = 0
    return csc_matrix(M)
    
    