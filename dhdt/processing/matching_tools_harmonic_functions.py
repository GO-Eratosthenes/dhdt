# general libraries
import numpy as np
from scipy import fftpack


# general frequency functions
def create_complex_DCT(I, C_c, C_s):  # todo
    C_cc, C_ss = C_c * I * C_c.T, C_s * I * C_s.T
    C_sc, C_cs = C_s * I * C_c.T, C_c * I * C_s.T

    C = C_cc - C_ss + 1j * (-(C_cs + C_sc))
    return C


def create_complex_fftpack_DCT(I):
    # DCT-based complex transform: {(C_cc - C_ss) -j(C_cs + C_sc)}
    C_cc = fftpack.dct(fftpack.dct(I, type=2, axis=0), type=2, axis=1)
    C_ss = fftpack.dst(fftpack.dst(I, type=2, axis=0), type=2, axis=1)
    C_cs = fftpack.dct(fftpack.dst(I, type=2, axis=0), type=2, axis=1)
    C_sc = fftpack.dst(fftpack.dct(I, type=2, axis=0), type=2, axis=1)

    C = (C_cc - C_ss) - 1j * (C_cs + C_sc)
    return C


def get_cosine_matrix(I, N=None):  # todo
    """ create matrix with a cosine-basis

    Parameters
    ----------
    I : numpy.array, size=(m,n)
        array with intensities
    N : integer
        amount of sine basis, if "None" is given "m" is used to create a square
        matrix

    Returns
    -------
    C : numpy.array, size=(m,N)
        cosine matrix

    See Also
    --------
    get_sine_matrix, create_complex_DCT, cosine_corr
    """
    (L, _) = I.shape
    if N == None:
        N = np.copy(L)
    C = np.zeros((L, L))
    for k in range(L):
        for n in range(N):
            if k == 0:
                C[k, n] = np.sqrt(
                    np.divide(1, L, out=np.zeros_like(L), where=L != 0))
            else:
                C[k, n] = np.sqrt(
                    np.divide(2, L, out=np.zeros_like(L), where=L != 0)) * \
                          np.cos(np.divide(np.pi * k * (1 / 2 + n), L,
                                           out=np.zeros_like(L), where=L != 0))
    return (C)


def get_sine_matrix(I, N=None):  # todo
    """ create matrix with a sine-basis

    Parameters
    ----------
    I : numpy.array, size=(m,n)
        array with intensities
    N : integer
        amount of sine basis, if "None" is given "m" is used to create a square
        matrix

    Returns
    -------
    C : numpy.array, size=(m,N)
        sine matrix

    See Also
    --------
    get_cosine_matrix, create_complex_DCT, cosine_corr
    """
    (L, _) = I.shape
    if N == None:
        # make a square matrix
        N = np.copy(L)
    C = np.zeros((L, L))
    for k in range(L):
        for n in range(N):
            if k == 0:
                C[k, n] = np.sqrt(
                    np.divide(1, L, out=np.zeros_like(L), where=L != 0))
            else:
                C[k, n] = np.sqrt(
                    np.divide(2, L, out=np.zeros_like(L), where=L != 0)) * \
                          np.sin(np.divide(np.pi * k * (1 / 2 + n), L,
                                           out=np.zeros_like(L), where=L != 0))
    return (C)
