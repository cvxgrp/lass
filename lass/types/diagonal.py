from lass.base import AtomicBlock
import numpy as np

class Diagonal(AtomicBlock):
    """ Storage for a diagonal matrix class.
    """
    def __init__(self, diag, k=0, *args, **kwargs):
        # k indicates which diagonal, 0 is the main, 1 is offset by 1 above,
        # -1 is offset by 1 below, etc.
        self.diag = diag    # entires along diagonal
        self.k = k          # kth diagonal
        n = len(self.diag)

        super(Diagonal, self).__init__(n, n, *args, **kwargs)

    @property
    def row_offset(self):
        if self.k <= 0: return 0
        else: return self.k

    @property
    def col_offset(self):
        if self.k >= 0: return 0
        else: return -self.k

    def __call__(self, x):
        tmp = np.zeros(self.rows)
        if self.k > 0:
            tmp[0:-self.k] = self.diag[0:-self.k]*x[self.k::]
        elif self.k < 0:
            tmp[-self.k::] = (self.diag*x)[0:self.k]
        else:
            tmp = self.diag * x
        return tmp


    def __getitem__(self, key):
        k1,k2 = key[0]-self.col_offset, key[1]-self.row_offset
        if k1 == k2:
            return self.diag[k1]
        else:
            return 0

    def __iter__(self):
        for k in xrange(abs(self.k),self.cols):
            i = k-self.row_offset
            j = k-self.col_offset
            v = self.diag[k-self.row_offset-self.col_offset]
            yield (i, j, v)