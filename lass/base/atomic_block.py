from abc import ABCMeta
from collections import Mapping, Callable

class AtomicBlock(Mapping, Callable):
    """ This is the base class that all blocks inherit from. It simply
        indicates that it has a size (rows,cols).

        All atomic blocks are mappings from tuple indices (i,j) to values v.
        They are also callable; calling the block applies its operation.

        The "length" of a block is equal to rows*cols; it assumes the matrix
        is dense.

        A Block contains a single AtomicBlock, but offset in some global
        mapping. A StructuredMatrix consists of a list of Blocks.
    """
    __metaclass__ = ABCMeta

    def __init__(self, rows, cols, *args, **kwargs):
        self.size = (rows, cols)
        self.rows = rows
        self.cols = cols
        super(AtomicBlock, self).__init__(*args, **kwargs)

    def __iter__(self):
        for i in xrange(self.rows):
            for j in xrange(self.cols):
                v = self[i,j]
                if v != 0: yield (i, j, v)

    def __len__(self):
        return self.rows*self.cols