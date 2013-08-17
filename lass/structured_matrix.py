from collections import Mapping, Callable
from lass.base import Block
import numpy as np


class StructuredMatrix(Mapping, Callable):
    def __init__(self, *args, **kwargs):
        self.blocks = []
        self.rows = 0
        self.cols = 0
        super(StructuredMatrix, self).__init__(*args, **kwargs)

    def push(self, block, row_start=0, col_start=0, row_stride=1, col_stride=1):
        new_block = Block(block, row_start, col_start, row_stride, col_stride)
        self.rows = max(self.rows, new_block.rows)
        self.cols = max(self.cols, new_block.cols)
        self.blocks.append(new_block)

    def __len__(self):
        return self.rows*self.cols

    def __call__(self,x):
        y = np.zeros(self.rows, dtype='complex')
        for b in self.blocks:
            y[b.row_start:b.rows:b.row_stride] += b(x[b.col_start:b.cols:b.col_stride])
        return y

    def __getitem__(self, key):
        # use map and reduce to get the value at a particular entry of the
        # structured matrix
        return sum(map(lambda x: x[key], blocks))

    def __iter__(self):
        for b in self.blocks:
            for i,j,v in b:
                yield (i,j,v)

