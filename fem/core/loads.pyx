class MassLoad:
    cpdef public double magnitude
    cpdef public int DOF
    
    def __cinit__(self, double magnitude, DOF):
        self.magnitude = magnitude
        self.DOF = DOF
    


cdef class NodalLoad:
    cpdef public double magnitude
    cpdef public int node, DOF
    
    def __cinit__(self, double magnitude, int node, int DOF):
        self.magnitude = magnitude
        self.node = node
        self.DOF = DOF


cdef class TimeDependentLoad(MassLoad):
    cpdef public double[:] _history
    
    def __init__(self, double[:] time_history, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.magnitude is None:
            self.magnitude = 1
        self._history = self.magnitude * time_history
        self.total_steps = len(time_history)
    
    cpdef double time_history(self, size_t timestep):
        if timestep < self.total_steps:
            return self._history[timestep]
        else:
            return 0

cdef class InertiaLoad(NodalLoad):
    
    
    
    def __init__(self, time_history=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.magnitude is None:
            self.magnitude = 1
        self._history = self.magnitude * time_history
        self.total_steps = len(time_history)
    
    def time_history(self, timestep):
        if timestep < self.total_steps:
            return self._history[timestep]
        else:
            return 0
    
    
    
    
    