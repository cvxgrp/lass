from collections import Mapping, Callable

class Block(Mapping, Callable):
    """ Wraps atomic blocks into an offset block
    """
    def __init__(self, atomic_block, row_start, col_start, row_stride = 1, col_stride = 1):
        self.block = atomic_block
        self.row_start = row_start
        self.col_start = col_start
        self.row_stride = row_stride
        self.col_stride = col_stride
        # set up initial size of block
        self.rows = self.row_start + self.block.rows*row_stride
        self.cols = self.col_start + self.block.cols*col_stride
        self.size = (self.rows, self.cols)
        super(Block, self).__init__()

    def __call__(self, x):
        return self.block(x)

    def __iter__(self):
        for i,j,v in self.block:
            yield (self.row_start + i*self.row_stride, self.col_start + j*self.col_stride, v)

    def __getitem__(self,key):
        i, j = key
        i -= self.row_start
        j -= self.col_start
        if i % self.row_stride == 0 and j % self.col_stride == 0:
            return self.f[i//self.row_stride, j//self.col_stride]
        else:
            return 0

    def __len__(self):
        return self.rows*self.cols