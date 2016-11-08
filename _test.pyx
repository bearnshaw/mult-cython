#cython: language_level=3
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True


import numpy as np
from cython.parallel import prange


def mult(double[:, :] a, double[:, :] b):
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


def mult_tensor(double[:, :] a, double[:, :] b):
    cdef:
        int i, j, k
        int I = a.shape[0]
        int J = a.shape[1]
        int K = b.shape[1]
        double[:, :, :] t = np.empty((I, J, K))
        double[:, :] c = np.zeros((I, K))

    assert(J == b.shape[0])

    for i in prange(I, nogil=True):
        for j in range(J):
            for k in range(K):
                t[i, j, k] = a[i, j] * b[j, k]
    for i in prange(I, nogil=True):
        for j in range(J):
            for k in range(K):
                c[i, k] += t[i, j, k]
    return np.asarray(c)


def mult_broken(double[:, :] a, double[:, :] b):
    """
    mult_broken is broken because the aggregation of c[i, k] is actually across
    threads, unlike in mult where the nontrivial summation is within a single
    thread.
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
