import numpy as np
import scipy.sparse as s
from lass.types import LinearOperator, Matrix, Diagonal, Eye
from lass.base import Block
from lass import StructuredMatrix

import time

def profile(f):
    def wrap(*args, **kwargs):
        start = time.clock()
        result = f(*args, **kwargs)
        elapsed = time.clock() - start
        print f.__name__, "took", elapsed, "secs"
        return result
    return wrap

@profile
def applyA(A, x):
    return A*x

@profile
def callSSM(A, x):
    return A(x)

@profile
def actual_op(x):
    return np.fft.fft(x[0:n]) + np.convolve(np.array([1,-2,1]), x[0:n],mode='same') + 2.1*x[n::]

@profile
def convert_to_coo(A):
    # somehow, this is faster than i,j,v = zip(*C)
    # but same as zip(*list(C))
    i,j,v = zip(*A.keys())
    return s.coo_matrix((v, (i,j)))


def f(x):
    return np.fft.fft(x,axis=0)

n = 1024

v = np.ones(n)
A = LinearOperator(f,rows=n,cols=n)  # general linear operator
B = Diagonal(v,1)
C = Diagonal(-2*v)
D = Diagonal(v,-1)
E = Eye(n,2.1)
# i may not actually need the "k" parameter in Diagonal
# just enter it into block with a row or column offset


#print list(B)

mat = StructuredMatrix()
mat.push(A)
mat.push(B)
mat.push(C)
mat.push(D)
mat.push(E,0,n)

# mat now does the following
# mat * [x,y] is
# take fft(x) + conv([1,-2,1],x) + 2.1*y

my_mat = convert_to_coo(mat)

x = np.random.randn(2*n)

y1 = applyA(my_mat, x)
y2 = actual_op(x)
y3 = callSSM(mat, x)

print "error from applying matrix", np.linalg.norm(y1-y2)
print "error from calling the actual op", np.linalg.norm(y2 - y2)
print "error from using structured matrix", np.linalg.norm(y3 - y2)


# getitem is effectively a O(N) operation (even for large matrices), a
# map-reduce
# call is a O(N) operation, where N is the number of blocks
# iter is a O(nnz) operation (iterates through all nonzeros in the matrix)



