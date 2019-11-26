#import cython_module as cm





cdef class PyClass:
    
    def __init__(self):
        pass
    
    



cdef class CyClass:
    cpdef public float x
    cdef PyClass myclass
    
    def __init__(self, float x, PyClass myclass):
        self.x = x
        self.myclass = myclass
    
    cpdef float foo(self, int n):
        cdef int i
        for i in range(n):
            self.x += 1
    
    cpdef boo(self):
        f = self.foo
        f(10)

cdef class Child(CyClass):
    
    def __init__(self, float x, PyClass myclass):
        super(CyClass, self).__init__(x, myclass)

cpdef  int foo( int n=10):
    cdef int x = 1
    cdef int i
    for i in range(n):
        x = x*n-i
    