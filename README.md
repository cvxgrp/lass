Linear Algebra for Structured Sparse Matrices
=============================================

This is a Python library for representing structured, sparse matrices. The
representation is compact and can be used in several ways:

* at run-time, to ensure the most efficient matrix-vector multiplication is
carried out

* as a method to distributed sparse matrices across clusters when used in
conjunction with PyTables

* as an intermediate representation for sparse linear algebra

It represents matrix structure *constructively*; it is not able to infer
matrix structure given a sparse matrix. Instead, it requires that the user
construct a sparse matrix using a set of primitive matrix structures.