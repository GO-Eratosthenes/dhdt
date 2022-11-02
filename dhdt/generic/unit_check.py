import numpy as np

def zenit_angle_check(zn):
    zn = np.maximum(zn, 0.)
    zn = np.minimum(zn, 90.)
    return zn


def are_two_arrays_equal(A, B):
    assert type(A) == np.ndarray, ('please provide an array')
    assert type(B) == np.ndarray, ('please provide an array')
    assert A.ndim==B.ndim, ('please provide arrays of same dimension')
    assert A.shape==B.shape, ('please provide arrays of equal shape')
    return

def are_three_arrays_equal(A, B, C):
    assert type(A) == np.ndarray, ('please provide an array')
    assert type(B) == np.ndarray, ('please provide an array')
    assert type(C) == np.ndarray, ('please provide an array')
    assert len(set({A.shape[0], B.shape[0], C.shape[0]})) == 1, \
         ('please provide arrays of the same size')
    assert len(set({A.shape[1], B.shape[1], C.shape[1]})) == 1, \
         ('please provide arrays of the same size')
    assert len(set({A.ndim, B.ndim, C.ndim})) == 1, \
         ('please provide arrays of the same size')
    return