



class PyClass:
    
    def __init__(self, n):
        self.n = n
    
    def foo(self):

        for i in range(10000):
            self.n += 1



cdef class CyClass:
    
    def __init__(self, float n,  myclass):
        self.n = n
    
    cdef int foo(self):
        cdef int i
        for i in range(10000):
            self.n += 1
    

    

cpdef  int foo( int n=10):
    cdef int x = 1
    cdef int i
    for i in range(n):
        x = x*n-i
    