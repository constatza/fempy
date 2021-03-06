 # -*- coding: utf-8 -*-
"""
Manifold-learning library

Created on Tue Nov 12 14:39:25 2019

@author: constatza
"""
from time import time
import numpy as np
import scipy.sparse.linalg as splinalg
import scipy.linalg as linalg

from dataclasses import dataclass, field
from sklearn.metrics.pairwise import pairwise_distances
from scipy.sparse import csr_matrix
from mathematics.stochastic import zscore

"""Dimensionality reduction classes"""


@dataclass
class Eigendecomposer:
    dataset : np.ndarray = field(default_factory=np.ndarray)
    
    def __post_init__(self):
        self.eigenvalues = None
        self.eigenvectors = None
        self.reduced_coordinates = None
    
    @property
    def znormed_eigenvalues(self):
        return zscore(self.eigenvalues)
    
    @property
    def znormed_eigenvectors(self):
        return zscore(self.eigenvectors)
        

@dataclass
class DiffusionMap(Eigendecomposer):
    
    epsilon : float = 1
    alpha : float = 1

    
    def fit(self, numeigs=1, t=1, inplace=True):

        eigenvalues, eigenvectors, coordinates = self.to_diffusion_coordinates(
                                                               numeigs=numeigs,
                                                               t=t)
        eigenvectors = eigenvectors
        coordinates = coordinates
        if inplace: 
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self.reduced_coordinates = coordinates
        else:
            return eigenvalues, eigenvectors, coordinates
    
    
    def kernel_sums_per_epsilon(self, dataset=None, epsilon=None, ax=None):
        """
            M(epsilon) = Sum(Sum(Lij))
            If epsilon is an array, loop over it to return an array.
        """
        if dataset is None: dataset = self.dataset
        kernel_matrix = DiffusionMap.calculate_kernel_matrix
        Me = np.empty(epsilon.shape)
        for i,e in enumerate(epsilon):
            L = kernel_matrix(dataset, epsilon=e)
            Me[i] = np.einsum('ij->', L)/Me.shape[0]
        return Me

    @staticmethod    
    def calculate_kernel_matrix(dataset, epsilon=1):
        distance_matrix = pairwise_distances(dataset.T, metric='euclidean')
        return np.exp(-distance_matrix/epsilon/epsilon)
    
    def to_diffusion_coordinates(self, dataset=None, numeigs=1, t=1, epsilon=None, alpha=None):
        if dataset is None: dataset = self.dataset
        if epsilon is None: epsilon = self.epsilon
        if alpha is None: alpha = self.alpha
        
        K = DiffusionMap.calculate_kernel_matrix(dataset, epsilon=epsilon)
        rowsums = np.sum(K, axis=1)
        D_a = np.power(rowsums, alpha)[:, np.newaxis]
        W = K / D_a / D_a.T
        
        D = np.sum(W, axis=1, keepdims=True)
        Droot = np.sqrt(D)
        self.transition_matrix = W/D
        # not the real markov transition matrix! 
        # symmetric analog just to use symmetric algorithm
        symmetric_markov = W / Droot / Droot.T 
    
        # # reltol = 1e-6
        # relmarkov = symmetric_markov/np.max(symmetric_markov)
        # symmetric_markov[relmarkov < reltol] = 0
        L = symmetric_markov.shape[0]    
        # eigenvalues, eigenvectors = linalg.eigh(symmetric_markov,
                                                       # eigvals=(L-numeigs-2, L-1))
        eigenvalues, eigenvectors = eigendecomposition(symmetric_markov, k = numeigs+1,                                           
                                                        which='LM',
                                                        timeit=True,
                                                        return_eigenvectors=True)
        
        eigenvectors = eigenvectors / Droot # real eigenvectors of markov matrix
        if t>1:
            eigenvalues = np.power(eigenvalues, t)
        Psi = eigenvalues[np.newaxis,:] * eigenvectors[:, :]
        
        return eigenvalues, eigenvectors[:, 1:].T, Psi.T

linalg.eigh
@dataclass
class PCA(Eigendecomposer):
    
    def __post_init__(self):
        self.eigenvectors = None
    
    def fit(self, numeigs=1, inplace=True):
        dataset = self.dataset - np.mean(self.dataset, axis=1, keepdims=True)
        m = dataset.shape[1]
        correl = dataset @ dataset.T/(m-1)
        eigenvalues, eigenvectors = eigendecomposition(correl, 
                                                       k=numeigs,
                                                       which='LM',
                                                       return_eigenvectors=True)
        
        eigenvectors = eigenvectors
        if inplace: 
            self.eigenvectors = eigenvectors.T
            self.eigenvalues = eigenvalues
            self.reduced_coordinates = eigenvectors.T @ dataset
            self.correl = correl
        else:
            return eigenvalues, eigenvectors
    


"""Maps"""

@dataclass
class Map:
    domain : np.ndarray = field(default_factory=np.ndarray)
    codomain : np.ndarray = field(default_factory=np.ndarray)


@dataclass
class LinearMap(Map):
    
    def __post_init__(self):
        try:
            p, res = LinearMap.transform(self.domain, self.codomain)
        except linalg.LinAlgError:
            p = None
            res = None
            print("error")
         
        self.matrix = p
        self.res = res
    
    @staticmethod
    def transform(domain, codomain):

        L = linalg.cho_factor(domain @ domain.T)
        linear_map_T = linalg.cho_solve(L, domain.dot(codomain.T))
        linear_map = linear_map_T.T
        diff = codomain - linear_map @ domain
        res = np.sum(diff*diff, axis=1)
        
        return linear_map, res
    
    def direct_transform_vector(self, vector: np.ndarray):
        return self.matrix.dot(vector)
    
    def transpose_transform_vector(self, vector: np.ndarray, *args, **kwargs):
        return self.matrix.T.dot(vector)
    
    def direct_transform_matrix(self, matrix):
        return self.matrix.T.dot(matrix.dot(self.matrix))
    
    def transpose_transform_matrix(self, matrix):
        return self.matrix.dot(matrix.dot(self.matrix.T))

def sparse_eigendecomposition(arrayh, M=None, timeit=False, **kwargs):            
    start = time()

    B = csr_matrix(arrayh)
    if M is not None:
        M = csr_matrix(M)
    if kwargs['return_eigenvectors']:
        eigenvalues, eigenvectors = splinalg.eigsh(B, M=M, **kwargs)
        eigenvectors =  np.flip(eigenvectors, axis=1)
        
    else:
        eigenvalues = splinalg.eigsh(B, M=M, **kwargs)

    eigenvalues = np.flip(eigenvalues)

    if timeit:
        end = time()
        print("Sparse Eigendecomposition in {:.2f} sec".format(end - start) )
    
    if kwargs['return_eigenvectors']:
        return eigenvalues, eigenvectors
    else:
        return eigenvalues
    

def matrix_density(matrix):
    length = matrix.shape[0]
    density = np.count_nonzero(matrix)/length**2
    return density
    
def dense_eigendecomposition(arrayh, M=None, timeit=False, **kwargs):
    start = time()
    N = arrayh.shape[0]

    k = kwargs['k']
    which = kwargs['which']
    
    if which[0]=='L':
        eigs = (N-k, N-1)
    else:
        eigs = (0, k)
        
    if kwargs['return_eigenvectors']:
        eigenvalues, eigenvectors = linalg.eigh(arrayh,
                                                b=M,
                                                eigvals=eigs,
                                                check_finite=False)
        eigenvectors =  np.flip(eigenvectors, axis=1)
        
    else:
        eigenvalues = linalg.eigh(arrayh,
                                  b=M,
                                  eigvals_only=True,
                                  eigvals=eigs,
                                  check_finite=False)

    eigenvalues = np.flip(eigenvalues)

    if timeit:
        end = time()
        print("Dense Eigendecomposition in {:.2f} sec".format(end - start) )
    
    if kwargs['return_eigenvectors']:
        return eigenvalues, eigenvectors
    else:
        return eigenvalues
    
def eigendecomposition(arrayh, M=None, timeit=False, critical_density=.9, **kwargs):
    den = matrix_density(arrayh)
    if den<critical_density and arrayh.shape[0]>100:
        return sparse_eigendecomposition(arrayh, M=M, timeit=timeit, **kwargs)
    else:
        return dense_eigendecomposition(arrayh, M=M, timeit=timeit, **kwargs)
        
        
    
    
    
# def nearest_neighbour_mapping(vectors, natural_coordinates, transformed_coordinates, k=3):
#      coordinates_tree = spatial.cKDTree(natural_coordinates)
     
#      distances, neighbours_id  = coordinates_tree.query(vectors, k=k)
     
#      d_inv = 1/distances
#      weights = d_inv/np.sum(d_inv, axis=1, keepdims=True)
#      weights[np.isnan(weights)] = 1
#      transformed_vectors = weights[:, :, None] * transformed_coordinates[neighbours_id, :]
#      transformed_vectors = np.sum(transformed_vectors, axis=1)
#      return transformed_vectors