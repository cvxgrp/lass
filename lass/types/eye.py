from lass.base import AtomicBlock

class Eye(AtomicBlock):
    """ Storage for the scaled identity matrix. Currently assumes
        numpy / scipy matrix.
    """
    def __init__(self, n, coeff=1, *args, **kwargs):
        self.coeff = coeff
        super(Eye, self).__init__(n, n, *args, **kwargs)

    def __iter__(self):
        for i in xrange(self.rows):
            yield (i, i, self.coeff)

    def __call__(self, x):
        return self.coeff*x

    def __getitem__(self, key):
        if key[0] == key[1]: return self.coeff
        else: return 0