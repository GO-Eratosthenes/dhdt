# general libraries
import numpy as np

# image processing libraries
from skimage.feature import match_template
from scipy import ndimage

# spatial pattern matching functions
def normalized_cross_corr(I1, I2):
    """ simple normalized cross correlation

    Parameters
    ----------
    I1 : np.array, type=bool
        binary array
    I2 : np.array, type=bool
        binary array

    Returns
    -------
    ccs : np.array
        similarity surface, ccs: cross correlation surface        
    """
    ccs = match_template(I2, I1)
    return ccs

def cumulative_cross_corr(I1, I2):
    """ doing normalized cross correlation on distance imagery

    Parameters
    ----------
    I1 : np.array, type=bool
        binary array
    I2 : np.array, type=bool
        binary array

    Returns
    -------
    ccs : np.array
        similarity surface, ccs: cross correlation surface     
    """   
    if isinstance(I1, np.floating):        
        # get cut-off value
        cu = np.quantile(I1, 0.5)
        I1 = I1<cu
    if isinstance(I1, np.floating):        
        # get cut-off value
        cu = np.quantile(I2, 0.5)
        I2 = I2<cu
        
    I1new = ndimage.distance_transform_edt(I1)
    I2new = ndimage.distance_transform_edt(I2)
    
    ccs = match_template(I2new, I1new)
    return ccs

def sum_sq_diff(I1, I2):
    """ sum of squared difference correlation

    Parameters
    ----------
    I1 : np.array
        image with intensities (template)
    I2 : np.array
        image with intensities (search space)

    Returns
    -------
    ssd : np.array
        dissimilarity surface, ssd: sum of squared differnce
    """
    
    t_size = I1.shape
    y = np.lib.stride_tricks.as_strided(I2,
                    shape=(I2.shape[0] - t_size[0] + 1,
                           I2.shape[1] - t_size[1] + 1,) +
                          t_size,
                    strides=I2.strides * 2)
    ssd = np.einsum('ijkl,kl->ij', y, I1)
    ssd *= - 2
    ssd += np.einsum('ijkl, ijkl->ij', y, y)
    ssd += np.einsum('ij, ij', I1, I1)
    return ssd

def sum_sad_diff(I1, I2):
    """ sum of absolute difference correlation

    Parameters
    ----------
    I1 : np.array
        image with intensities (template)
    I2 : np.array
        image with intensities (search space)

    Returns
    -------
    sad : np.array
        dissimilarity surface, sad: sum of absolute differnce
    """
    
    t_size = I1.shape
    y = np.lib.stride_tricks.as_strided(I2,
                    shape=(I2.shape[0] - t_size[0] + 1,
                           I2.shape[1] - t_size[1] + 1,) +
                          t_size,
                    strides=I2.strides * 2)
    sad = np.einsum('ijkl,kl->ij', y, I1)
    return sad

# sum of robust differences, see Li_03
# least squares matching