# general libraries
import numpy as np
from scipy import fftpack


# general frequency functions
def create_complex_DCT(Z, C_c, C_s):  # todo
    C_cc, C_ss = C_c * Z * C_c.T, C_s * Z * C_s.T
    C_sc, C_cs = C_s * Z * C_c.T, C_c * Z * C_s.T

    C = C_cc - C_ss + 1j * (-(C_cs + C_sc))
    return C


def create_complex_fftpack_DCT(Z):
    # DCT-based complex transform: {(C_cc - C_ss) -j(C_cs + C_sc)}
    C_cc = fftpack.dct(fftpack.dct(Z, type=2, axis=0), type=2, axis=1)
    C_ss = fftpack.dst(fftpack.dst(Z, type=2, axis=0), type=2, axis=1)
    C_cs = fftpack.dct(fftpack.dst(Z, type=2, axis=0), type=2, axis=1)
    C_sc = fftpack.dst(fftpack.dct(Z, type=2, axis=0), type=2, axis=1)

    C = (C_cc - C_ss) - 1j * (C_cs + C_sc)
    return C


def get_cosine_matrix(Z, N=None):  # todo
    """ create matrix with a cosine-basis

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
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
    (L, _) = Z.shape
    if N is None:
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


def get_sine_matrix(Z, N=None):  # todo
    """ create matrix with a sine-basis

    Parameters
    ----------
    Z : numpy.array, size=(m,n)
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
    (L, _) = Z.shape
    if N is None:
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
