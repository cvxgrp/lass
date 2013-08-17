from lass.base import AtomicBlock

class Matrix(AtomicBlock):
    """ Storage for a matrix class. Currently assumes numpy / scipy matrix.
    """
    def __init__(self, A, *args, **kwargs):
        self.A = A
        m, n = A.shape

        super(Matrix, self).__init__(m, n, *args, **kwargs)

    def __call__(self, x):
        return self.A*x

    def __getitem__(self, key):
        return self.A[key]

