import numpy as np

from dhdt.generic.test_tools import create_sample_image_pair

def sum_sad_diff(I1, I2):
    """ sum of absolute difference correlation

    Parameters
    ----------
    I1 : numpy.array
        image with intensities (template)
    I2 : numpy.array
        image with intensities (search space)

    Returns
    -------
    sad : numpy.array
        dissimilarity surface, sad: sum of absolute difference
    """

    t_size = I1.shape
    y = np.lib.stride_tricks.as_strided(I2,
                                        shape=(I2.shape[0] - t_size[0] + 1,
                                               I2.shape[1] - t_size[1] + 1,) +
                                              t_size,
                                        strides=I2.strides * 2)
    sad = np.einsum('ijkl,kl->ij', y, I1)
    return sad

def norm_inner_prod(I1, I2):
    """ sum of absolute difference correlation

    Parameters
    ----------
    I1 : numpy.array, ndim=3
        image with intensities (template)
    I2 : numpy.array, ndim=3
        image with intensities (search space)

    Returns
    -------
    sad : numpy.array
        dissimilarity surface, sad: sum of absolute difference

    References
    ----------
    .. [1] Wang et al. "Landslide displacement monitoring by a fully
           Polarimetric SAR offset tracking method" Remote sensing, vol.8
           pp.624, 2016.
    """

    t_size = I1.shape
    y = np.lib.stride_tricks.as_strided(I2,
                                        shape=(I2.shape[0] - t_size[0] + 1,
                                               I2.shape[1] - t_size[1] + 1,
                                               I2.shape[2]) +
                                              t_size,
                                        strides=I2.strides * 3)
    nip = np.einsum('ijklmn,klmn->ijk', y, I1)
    return nip

_,I_2,_,_,I_1 = create_sample_image_pair(d=2**7, max_range=1,
                                         integer=False, ndim=3)
C = sum_sad_diff(np.squeeze(I_2[...,0]), np.squeeze(I_1[...,0]))
C_I = norm_inner_prod(I_2,I_1)

print('.')
C = sum_sad_diff(np.squeeze(I_2[...,0]), np.squeeze(I_1[...,0]))
