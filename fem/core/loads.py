class MassLoad:
    
    def __init__(self, magnitude=None, DOF=None):
        self.magnitude = magnitude
        self.DOF = DOF
    


class NodalLoad:
    
    def __init__(self, magnitude=None, node=None, DOF=None):
        self.magnitude = magnitude
        self.node = node
        self.DOF = DOF


class TimeDependentLoad(NodalLoad):
    
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

class InertiaLoad(MassLoad):
    
    def __init__(self, time_history=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.magnitude is None:
            self.magnitude = 1
        self._history = self.magnitude * time_history
        self.total_steps = time_history.shape[0]
    
    def time_history(self, timestep, direction_vector, out):
        if timestep < self.total_steps:
             out += self._history[timestep] * direction_vector
        else:
             pass
    
    
    
    
    