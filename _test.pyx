#cython: language_level=3
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True


import numpy as np
from cython.parallel import prange


def mult(double[:, :] a, double[:, :] b):
    """
    mult works because the entry c[i, k] is calculated on a single thread, so
    the reduction is trivial.
    """
    cdef:
        int i, j, k, ik
        int I = a.shape[0]
        int J = a.shape[1]
        int K = b.shape[1]
        double[:, :] c = np.zeros((I, K))

    assert(J == b.shape[0])

    for ik in prange(I * K, nogil=True):
        i = ik / K
        k = ik % K
        for j in range(J):
            c[i, k] += a[i, j] * b[j, k]
    return np.asarray(c)


def mult_broken(double[:, :] a, double[:, :] b):
    """
    mult_broken does not work because the entry c[i, k] is calculated across
    multiple threads and thus requires a non-trivial reduction.
    """
    cdef:
        int i, j, k, ik
        int I = a.shape[0]
        int J = a.shape[1]
        int K = b.shape[1]
        double[:, :] c = np.zeros((I, K))

    assert(J == b.shape[0])

    for j in prange(J, nogil=True):
        for ik in range(I * K):
            i = ik / K
            k = ik % K
            c[i, k] += a[i, j] * b[j, k]
    return np.asarray(c)
