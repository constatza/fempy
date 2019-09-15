# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:03:35 2019

@author: constatza
"""

import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as linalg
import scipy.optimize as opt
import numba as nb
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
    def data_set(self, value):
        self._diffusion_matrix = None
        self._dataset = value
    
    @property
    def diffusion_matrix(self):
        if self._diffusion_matrix is None:
            self._diffusion_matrix = self.calculate_diffusion_matrix(self.data_set, 
                                                                     epsilon=self.epsilon)
        return self._diffusion_matrix
    
    @property
    def transition_matrix(self):
        return self.calculate_transition_matrix(self.data_set, episolon=self.epsilon)


def M(data_set, epsilon):
        """
            M(epsilon) = Sum(Sum(Lij))
            If epsilon is an array, loop over it to return an array.
        """
        if type(epsilon)==np.ndarray:
            Me = np.empty(epsilon.shape)
            for i,e in enumerate(epsilon):
                Me[i] = calculate_M(data_set, epsilon=e)
            return Me
        else:
            
            Me = calculate_M(data_set, epsilon=epsilon)
            return Me    


def calculate_M(data_set, epsilon=1):
    L = calculate_diffusion_matrix(data_set, epsilon=epsilon)
    return np.sum(np.sum(L, axis=1))


@nb.njit
def calculate_diffusion_matrix(data_set, epsilon=1):
    numdata = data_set.shape[1]
    deltaU = np.zeros((numdata, numdata))
    for i in range(numdata):
        for j in range(numdata):
            for k in range(data_set.shape[0]):
                diff = data_set[k, i] - data_set[k, j]
                deltaU[i, j] += diff*diff
    k = np.exp(-deltaU/epsilon/epsilon)
    
    return k 
    

def calculate_transition_matrix(matrix):
    """ Converts the input matrix to transition matrix
        
    T = D**(-1) @ matrix
    
        where:
        Dii component = sum(matrix, axis=1)
    """
    
    rowsums = np.sum(matrix, axis=1)
    T = ( 1/rowsums[:, np.newaxis] ) * matrix
    return T


def diffusion_maps(data_set, numeigs=10, t=1, epsilon=None):
    k = numeigs
    L = calculate_diffusion_matrix(data_set, epsilon=epsilon)
    L [ abs(L)<1e-12] = 0
    P = calculate_transition_matrix(L)
    P[np.abs(P)<1e-12] = 0
    if t>1:
        P = np.linalg.matrix_power(P,t)
    eigenvalues, eigenvectors = linalg.eigsh(P, k=k, which='LM')
    ordered = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[ordered]
    eigenvectors = eigenvectors[:, ordered]
    eigenvectors[np.abs(eigenvectors)<1e-12] = 0
    return eigenvalues, eigenvectors
 

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
        lhs[ np.abs(lhs)<1e-12] = 0
        lhs = csr_matrix(lhs)
        transformation_matrix = linalg.spsolve(lhs, rhs)
        
    else:#if lhs is scalar
        transformation_matrix = rhs/lhs
    diff = natural_coordinates - transformation_matrix.T @ diffusion_coordinates.T
    res =  np.sum(np.sum(diff*diff))
    return transformation_matrix.T, res


def func(x, U, Z):
    d = U.shape[0]
    n = Z.shape[1]
    
    A = x.reshape((d, n))
    
    res = U - A @ Z.T
    return res.ravel()
    
    
def nl_least_squares(natural_coordinates, diffusion_coordinates):
    
    
    x0 = ls_approx(natural_coordinates=natural_coordinates, 
                                 diffusion_coordinates=diffusion_coordinates)
    x0 = x0.ravel()
    

    res = opt.least_squares(func, x0, args=(natural_coordinates, diffusion_coordinates))
    
    return res.x.reshape((natural_coordinates.shape[0], diffusion_coordinates.shape[1]))
        
  
if __name__=='__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    
    epsilon = 1

    timesteps = 10
    numeigs = 12
    
    sigma = 0.05
    t = np.linspace(0, 3.5*np.pi, 200)
    x = np.cos(t)#*(1 + np.random.normal(scale=sigma, size=len(t)) )
    y = np.sin(t)#*(1 + np.random.normal(scale=sigma, size=len(t)) )
    z = t*t#*(1 + np.random.normal(scale=sigma, size=len(t)) )
    U = np.concatenate([[x],[y],[z]])

    
    e = np.logspace(-3,3)
    
    eigvals, eigvecs = diffusion_maps(U, epsilon=epsilon, t=timesteps, numeigs=numeigs)
    Fi = eigvals[:] * eigvecs[:, :]
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    cumsum = M(U, epsilon=e)
    plt.figure()  
    plt.loglog(e, cumsum)
    
    plt.figure()
    plt.plot(eigvals, '.-')
    plt.ylabel('eigenvalues')
    plt.grid()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(Fi[:,1], Fi[:,2], Fi[:,3])
    plt.grid()
    plt.ylabel('Ψ2')
    plt.xlabel('Ψ1')
    plt.show() 
    k = k = len(eigvals[eigvals>0.05])
    A,res = ls_approx(U, Fi[:,:k])
#    A = nl_least_squares(U, Fi[:, :k])
    print(A.shape)
    Unew = A @ Fi[:,:k].T
    xnew = Unew[0,:]
    ynew = Unew[1,:]
    znew = Unew[2,:]
    ax.scatter(xnew,ynew,znew, color='g')     

  