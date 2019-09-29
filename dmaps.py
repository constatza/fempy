# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 18:03:35 2019

@author: constatza
"""

import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as splinalg
import scipy.optimize as opt
import scipy.linalg as linalg
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
                L = calculate_diffusion_matrix(data_set, epsilon=e)
                Me[i] = np.einsum('ij->', L)
           
        else:
            L = calculate_diffusion_matrix(data_set, epsilon=e)
            Me = np.einsum('ij->', L)
        return Me    


@nb.njit('float64[:, :](float64[:, :], float64)', parallel=True)
def calculate_diffusion_matrix(data_set, epsilon=1):

    numdata = data_set.shape[1]
    deltaU = np.zeros((numdata, numdata))
    K = np.empty((numdata, numdata))
    for i in nb.prange(numdata):
        for j in range(i, numdata):
            for k in nb.prange(data_set.shape[0]):
                diff = data_set[k, i] - data_set[k, j]
                deltaU[i, j] += diff * diff
                
            deltaU[j, i] = deltaU[i, j]
            
    K = np.exp(-deltaU/epsilon/epsilon)
    
    return K
    

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
    L = calculate_diffusion_matrix(data_set, epsilon=epsilon)
    P = calculate_transition_matrix(L)

    if t>1:
        P = np.linalg.matrix_power(P,t)

    B = csr_matrix(P)
    
    eigenvalues, eigenvectors = splinalg.eigs(B, k=numeigs, which='LR')
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real
    
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
        lhs[ np.abs(lhs)<1e-15] = 0
        lhs = csr_matrix(lhs)
        transformation_matrix = splinalg.spsolve(lhs, rhs)
        
    else:#if lhs is scalar
        transformation_matrix = rhs/lhs
    diff = natural_coordinates - transformation_matrix.T @ diffusion_coordinates.T
    res =  np.sum(np.sum(diff*diff, axis=1))
    return transformation_matrix.T, res


def func(x, U, Z):
    d = U.shape[0]
    n = Z.shape[1]
    A = x[:d*n].reshape((d, n))
    B = x[d*n:].reshape((d, n))
    res = U - A @ Z.T - B @ Z.T**3
    return res.ravel()
    
    
def nl_least_squares(natural_coordinates, diffusion_coordinates):
    
#    x0, res  = ls_approx(natural_coordinates=natural_coordinates, 
#                                 diffusion_coordinates=diffusion_coordinates)
#    x0 = x0.ravel()
    d = natural_coordinates.shape[0]
    n = diffusion_coordinates.shape[1]
    x0 = np.random.rand(2*d*n)
    
    res = opt.least_squares(func, x0, args=(natural_coordinates, diffusion_coordinates))
    
    return res.x

def normalize(U):
    try:
        Umean = np.mean(U, axis=1, keepdims=True)
        Ustd = np.std(U, axis=1, keepdims=True)
    except IndexError:
        Umean = np.mean(U, keepdims=True)
        Ustd = np.std(U, keepdims=True)
    
    Unormalized = ( U - Umean)/Ustd
    return Unormalized, Umean, Ustd

def denormalize(U, Umean, Ustd):
    return U*Ustd + Umean

def pca(data_set, numeigs):
    correl = data_set.T @ data_set
    val, vec = splinalg.eigsh(correl, k=numeigs, which='LM')
    val = val[::-1]
    vec = vec[:, ::-1]  
    return val, vec


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    plt.close('all')
    
    epsilon = .5

    timesteps = 500
    numeigs = 2
    
    sigma = 0.05
    t = np.linspace(0, 2*np.pi, 100)
    x = np.cos(t)*(1 + np.random.normal(scale=sigma, size=len(t)) )
    y = np.sin(t)*(1 + np.random.normal(scale=sigma, size=len(t)) )
    z = t*(1 + np.random.normal(scale=sigma, size=len(t)) )
    U = np.concatenate([[x],[y],[z]])
    U, Umean, Ustd = normalize(U)
    d = U.shape[0]
    N = U.shape[1]
    
    """Diffusion Maps"""   
    e = np.logspace(-3,3, num=20)
    
    eigvals, eigvecs = diffusion_maps(U, epsilon=epsilon, t=timesteps, numeigs=numeigs+1)
    #k = len(eigvals[eigvals>0.05]) +1
    Fi =  eigvecs[:, :]
    
    sol = nl_least_squares(U, Fi)
    d = U.shape[0]
    n = Fi.shape[1]
    A = sol[:d*n].reshape((d, n))
    B = sol[d*n:].reshape((d, n))
    res = U - A @ Fi.T - B @ Fi.T**3
    res = np.einsum('ij->', res*res)
    print(A.shape)
    Unew = denormalize(A @ Fi.T + B @ Fi.T**3, Umean, Ustd)
    xnew = Unew[0,:]
    ynew = Unew[1,:]
    znew = Unew[2,:]
    
    """PCA"""
    val, vec = pca(U, numeigs=numeigs)
    m = len(val[val>[.05]]) 
    Lr = vec[:, :m]
    P, res2 = ls_approx(U, Lr) 

    U_new2 = denormalize(P @ Lr.T, Umean, Ustd) 
    x_pca = U_new2[0, :]
    y_pca = U_new2[1, :]
    z_pca = U_new2[2, :]
    
    """Plots"""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, alpha=0.5, label='Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    cumsum = M(U, epsilon=e)
    plt.figure()  
    plt.loglog(e, cumsum)
    
    plt.figure()
    plt.plot(eigvals, '.-', label='Diffusion Maps')
    plt.plot(val/np.max(val), '+-', label='PCA')
    plt.ylabel('eigenvalues')
    plt.grid()
    
    
    
    ax.scatter(xnew,ynew,znew, color='g', label='Diffusion Maps')  
    ax.scatter(x_pca,y_pca,z_pca, color='r', label='PCA')
    ax.legend()
  
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    try:
        ax2.scatter(Fi[:,1], Fi[:,2], Fi[:,3])
        ax2.scatter(Lr[:,0], Lr[:,1], Lr[:,2])
    except IndexError:
        pass
    plt.grid()
    plt.ylabel('$\Psi_2$')
    plt.xlabel('$\Psi_1$')
    plt.show() 