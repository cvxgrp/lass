from lass.base import AtomicBlock
import numpy as np

class LinearOperator(AtomicBlock):
    """An opaque linear operator. Assumed to be dense."""
    def __init__(self,f, *args, **kwargs):
        self.f = f
        super(LinearOperator, self).__init__(*args, **kwargs)

    def __call__(self, x):
        return self.f(x)

    def __getitem__(self, key):
        # for a generic SSM block
        assert(0 <= key[0] < self.rows)
        assert(0 <= key[1] < self.cols)
        # send e_i to the operator
        e = np.zeros(self.cols)
        e[key[1]] = 1
        y = self(e)
        return y[key[0]]