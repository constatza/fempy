import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cpdef void cholesky_factor(double[:, ::1] A):
    """cholesky factorization of real symmetric positive definite float matrix A

    Parameters
    ----------
    A : memoryview (numpy array)
        n x n matrix to compute cholesky decomposition
    B : memoryview (numpy array)
        n x n matrix to use within function, will be modified
        in place to become cholesky decomposition of A. works
        similar to np.linalg.cholesky
    """
    cdef double[:, ::1] L
    
    cdef Py_ssize_t n = A.shape[0]
    cdef size_t i, j
    
    for j in range(n):
        for i in range(n):
            if i>j:
                pass
                
                
potrs, = get_lapack_funcs(('potrs',), (c, b1))
    x, info = potrs(c, b1, lower=lower, overwrite_b=overwrite_b)
