# -*- coding: utf-8 -*-
"""
Manifold-learning library

Created on Tue Nov 12 14:39:25 2019

@author: constatza
"""
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
    
    @staticmethod
    def znormed_eigenvalues(eigenvalues):
        return zscore(eigenvalues)
    
    @staticmethod
    def znormed_eigenvectors(eigenvectors):
        return zscore(eigenvectors)
        

@dataclass
class DiffusionMap(Eigendecomposer):
    
    epsilon : float = 1
    alpha : float = 1

    
    def fit(self, numeigs=1, t=1, inplace=True):

        eigenvalues, eigenvectors, coordinates = DiffusionMap.to_diffusion_coordinates(self.dataset,
                                                               numeigs=numeigs,
                                                               t=t,
                                                               epsilon=self.epsilon,
                                                               alpha=self.alpha)
        eigenvectors = self.znormed_eigenvectors(eigenvectors)
        coordinates = self.znormed_eigenvectors(coordinates)
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
            Me[i] = np.einsum('ij->', L)
        return Me

    @staticmethod    
    def calculate_kernel_matrix(data_set, epsilon=1):
        distance_matrix = pairwise_distances(data_set.T, metric='euclidean')
        return np.exp(-distance_matrix/epsilon/epsilon)
    
    @staticmethod
    def to_diffusion_coordinates(data_set, numeigs=1, t=1, epsilon=1, alpha=0):
        
        K = DiffusionMap.calculate_kernel_matrix(data_set, epsilon=epsilon)
        rowsums = np.sum(K, axis=1)
        D_a = np.power(rowsums, alpha)[:, np.newaxis]
        W = K / D_a / D_a.T
        
        D = np.sum(W, axis=1, keepdims=True)
        Droot = np.sqrt(D)
        #P = W/D
        # not the real markov transition matrix! 
        # symmetric analog just to use symmetric algorithm
        symmetric_markov = W / Droot / Droot.T 
    
        
        
        reltol = 1e-4
        relmarkov = symmetric_markov/np.max(symmetric_markov)
        symmetric_markov[relmarkov < reltol] = 0
    
        B = csr_matrix(symmetric_markov)
        eigenvalues, eigenvectors = splinalg.eigsh(B, k=numeigs+1, which='LM')
        eigenvalues = np.flip(eigenvalues)  
        eigenvectors =  np.flip(eigenvectors, axis=1)
        eigenvectors = eigenvectors / Droot # real eigenvectors of markov matrix
        if t>1:
            eigenvalues = np.power(eigenvalues, t)
        Psi = eigenvalues[np.newaxis,:] * eigenvectors[:, :]
        
        return eigenvalues, eigenvectors[:, 1:].T, Psi.T


@dataclass
class PCA(Eigendecomposer):
    
    def __post_init__(self):
        self.eigenvectors = None
    
    def fit(self, numeigs=1, inplace=True):
        eigenvalues, eigenvectors = PCA.eigendecomposition(self.dataset, numeigs=numeigs)
        eigenvectors = self.znormed_eigenvectors(eigenvectors)
        if inplace: 
            self.eigenvectors = eigenvectors
            self.eigenvalues = eigenvalues
            self.reduced_coordinates = eigenvectors
        else:
            return eigenvalues, eigenvectors
    
    @staticmethod
    def eigendecomposition(dataset, numeigs=1):
        correl = dataset.T @ dataset
        val, vec = splinalg.eigsh(correl, k=numeigs, which='LM')
        val = val[::-1]
        vec = vec[:, ::-1]  
        return   val, vec.T


"""Maps"""

@dataclass
class Map:
    domain : np.ndarray = field(default_factory=np.ndarray)
    codomain : np.ndarray = field(default_factory=np.ndarray)


@dataclass
class LinearMap(Map):
    
    def __post_init__(self):
        
        p, res = LinearMap.transform(self.domain, self.codomain)

         
        self.matrix = p
        self.res = res
    
    @staticmethod
    def transform(domain, codomain):

        L = linalg.cho_factor(domain @ domain.T)
        linear_map_T = linalg.cho_solve(L, domain @ codomain.T)
        linear_map = linear_map_T.T
        diff = codomain - linear_map @ domain
        res = np.sum(diff*diff, axis=1)
        
        return linear_map, res
    
    def direct_transform_vector(self, vector: np.ndarray):
        return self.matrix @ vector
    
    def transpose_transform_vector(self, vector: np.ndarray, *args, **kwargs):
        return self.matrix.T @ vector
    
    def direct_transform_matrix(self, matrix):
        return self.matrix.T @ matrix @ self.matrix
    
    def transpose_transform_matrix(self, matrix):
        return self.matrix @ matrix @ self.matrix.T
            

# def nearest_neighbour_mapping(vectors, natural_coordinates, transformed_coordinates, k=3):
#      coordinates_tree = spatial.cKDTree(natural_coordinates)
     
#      distances, neighbours_id  = coordinates_tree.query(vectors, k=k)
     
#      d_inv = 1/distances
#      weights = d_inv/np.sum(d_inv, axis=1, keepdims=True)
#      weights[np.isnan(weights)] = 1
#      transformed_vectors = weights[:, :, None] * transformed_coordinates[neighbours_id, :]
#      transformed_vectors = np.sum(transformed_vectors, axis=1)
#      return transformed_vectors