# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:03:35 2019

@author: constatza
"""

import numpy as np
import scipy.linalg as linalg


class DiffusionMaps:
    
    def __init__(self, data_set=None, epsilon=None):
        """
        Parameters
        ----------
        data_set : numpy.ndarray
            Array with of shape (d, n). Each column represents a vector in R^d.
                d : number of euclidean space dimensions
                n : number of data or vectors
        """
        self._data_set = data_set
        self.number_of_dimensions = data_set.shape[0]
        self.number_of_data = data_set.shape[1]
        self._diffusion_matrix = None
        self.epsilon = epsilon
    
    @property
    def data_set(self):
        return self._data_set
    
    @data_set.setter
    def data_set(self, other):
        self._diffusion_matrix = None
        self._dataset = other
    
    def M(data_set, epsilon):
        """
            M(epsilon) = Sum(Sum(Lij))
            If epsilon is an array, loop over it to return an array.
        """
        if type(epsilon)==np.ndarray:
            Me = np.empty(epsilon.shape)
            for i,e in enumerate(epsilon):
                Me[i] = DiffusionMaps.calculate_M(data_set, epsilon=e)
            return Me
        else:
            
            Me = DiffusionMaps.calculate_M(data_set, epsilon=epsilon)
            return Me

    @property
    def diffusion_matrix(self):
        if self._diffusion_matrix is None:
            self._diffusion_matrix = self.calculate_diffusion_matrix(self.data_set, 
                                                                     epsilon=self.epsilon)
        return self._diffusion_matrix
    
    @property
    def transition_matrix(self):
        return self.calculate_transition_matrix(self.data_set, episolon=self.epsilon)
    
    @staticmethod
    def exponential_kernel(norm_squared, epsilon=1):
        return np.exp(-norm_squared/epsilon/epsilon)
    
    @staticmethod
    def calculate_M(data_set, epsilon=1):
        L = DiffusionMaps.calculate_diffusion_matrix(data_set, epsilon=epsilon)
        return np.einsum('ij->', L)
    
    @staticmethod
    def calculate_diffusion_matrix(data_set, epsilon=1):
        d = data_set.shape[0]
        n = data_set.shape[1]
        deltaU = np.empty((n, n, d))
        for k in range(d):
                ui = data_set[k, :] 
                uj = ui[:, np.newaxis]
                deltaU[:, :, k] = ui - uj
        
        distance_ij = np.einsum('ijk,ijk-> ij', deltaU, deltaU)
        return DiffusionMaps.exponential_kernel(distance_ij, epsilon=epsilon)
        
    @staticmethod
    def calculate_transition_matrix(data_set, epsilon=1):
        """ Computes the matrix T = D**(-1) @ matrix, where
        Dii component = sum(matrix, axis=1)
        """
        L = DiffusionMaps.calculate_diffusion_matrix(data_set, epsilon=epsilon)
        D = np.diag(np.einsum('ij->i', L))
        return linalg.solve(D, L)

    @staticmethod
    def diffusion_maps(data_set, epsilon=None, t=1, k=10):
        P = DiffusionMaps.calculate_transition_matrix(data_set, epsilon=epsilon)
        N = P.shape[0]
        P = np.linalg.matrix_power(P,t)
        eigenvalues, eigenvectors = linalg.eigh(P, eigvals=(N-k,N-1))
        ordered = np.argsort(-eigenvalues)
        return eigenvalues[ordered], eigenvectors[:, ordered]
     
    @staticmethod
    def ls_approx(natural_coordinates, diffusion_coordinates):
        """Least squares, linear approximation of the generally nonlinear
        transformation matrix from sample space to diffusion space.
        
        ui = A zi => 
        
        U = A Z.T => 
        
        A = U Z (Z.T Z)^(-1) 
        
        A.T = (Z.T Z)^(-T) Z.T U.T
        
        U: natural coordinates dataset, d x N 
        A: tranformation matrix, d x n
        Z: diffusion coordinates dataset,  N x n
        
        Parameters:
        -----------
        natural_coordinates : numpy.ndarray
            Dataset in natural (original) coordinates with size d x N
            d : number of dimensions in original space
            N ; number of data
        diffusion_coordinates : numpy.ndarray
            Dataset in diffusion coordinates with size N x n
            N : number of data
            n : number of dimensions in diffusion space
        
        Returns:
        --------
        tranformation_matrix : numpy.ndarray
            Linear (least squares) approximation of transformation matrix from
            original to diffusion maps space with size d x N.
        """
        
        rhs = diffusion_coordinates.T @ natural_coordinates.T
        lhs = diffusion_coordinates.T @ diffusion_coordinates
        if lhs.shape!=():
            L = linalg.cho_factor(lhs)
            transformation_matrix = linalg.cho_solve(L, rhs)
            
        else:#if lhs is scalar
            transformation_matrix = rhs/lhs
        
        return transformation_matrix.T
    
    
    def vectorize(func):
        
        pass
  
if __name__=='__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    
    epsilon = 1
    timesteps = 1
    numeigs = 10
    
    t = np.linspace(0, 4*np.pi, 500)
    x = np.cos(t) #+ np.random.normal(scale=0.05, size=len(t))
    y = np.sin(t) #+ np.random.normal(scale=0.05, size=len(t))
    z = t
    U = np.concatenate([[x],[y],[z]])

    DM = DiffusionMaps
    e = np.logspace(-3,3)
    
    eigvals, eigvecs = DM.diffusion_maps(U, epsilon=epsilon, t=timesteps, k=numeigs)
    Fi = np.exp(-eigvals) * eigvecs
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

  
    plt.figure()  
    plt.loglog(e, DM.M(U, epsilon=e))
    
    plt.figure()
    plt.plot(eigvals[:12], 'o-')
    plt.ylabel('eigenvalues')
    plt.grid()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(Fi[:,1], Fi[:,2], Fi[:,3])
    plt.grid()
    plt.ylabel('Ψ2')
    plt.xlabel('Ψ1')
    plt.show() 
    k = 8
    A = DM.ls_approx(U, Fi[:,:k])
    
    print(A.shape)
    Unew = A @ Fi[:,:k].T
    xnew = Unew[0,:]
    ynew = Unew[1,:]
    znew = Unew[2,:]
    ax.scatter(xnew,ynew,znew, color='g')     

