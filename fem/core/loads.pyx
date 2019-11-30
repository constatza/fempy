# cython : language_level=3
cimport cython
import numpy as np 
cimport numpy as np

cdef class MassLoad:
    cpdef public double magnitude
    cpdef public object DOF
    
    def __init__(self, double magnitude=1, DOF=None):
        self.magnitude = magnitude
        self.DOF = DOF
    


cdef class NodalLoad:
    cpdef public double magnitude
    cpdef public object DOF
    
    def __init__(self, double magnitude=1, node=None, DOF=None):
        self.magnitude = magnitude
        self.node = node
        self.DOF = DOF


cdef class TimeDependentLoad(NodalLoad):
    cdef double[::1] _history
    cdef int total_steps
    
    def __init__(self, time_history=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.magnitude is None:
            self.magnitude = 1
       
        self._history = time_history
        cdef size_t i
        cdef double magnitude = self.magnitude
        for i in range(time_history.shape[0]):
            self._history[i] *= magnitude 
        self.total_steps = len(time_history)
    
#    @cython.boundscheck(False)  # Deactivate bounds checking
#    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef double time_history(self, size_t timestep):
        if timestep < self.total_steps:
            return self._history[timestep]
        else:
            return 0



cdef class InertiaLoad(MassLoad):
    cdef double[::1] _history
    cdef int total_steps
    
    def __init__(self, time_history=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.magnitude is None:
            self.magnitude = 1
       
        self._history = time_history
        cdef size_t i
        cdef double magnitude = self.magnitude
        for i in range(time_history.shape[0]):
            self._history[i] *= magnitude 
        self.total_steps = time_history.shape[0]
    
    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef void time_history(self, size_t timestep,
                            np.ndarray[np.float64_t, ndim=1] direction_vector,
                            np.ndarray[np.float64_t, ndim=1] out):
        if timestep < self.total_steps:
            out += self._history[timestep] * direction_vector
        else:
            pass
    
    
    

    
    
    
    