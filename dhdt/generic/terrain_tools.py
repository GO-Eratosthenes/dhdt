import numpy as np

from scipy.ndimage import convolve #, filters

from .handler_im import get_grad_filters
from .filtering_statistical import make_2D_Laplacian

def terrain_curvature(Z):
#    C = filters.laplace(Z)
    l = make_2D_Laplacian(alpha=.5)
    C = convolve(Z,-l)
    return C

def ridge_orientation(Z): #todo: check if correct
    dx2,_ = get_grad_filters(ftype='sobel', tsize=3, order=2)
    dZ_dx2 = convolve(Z,-dx2)
    dy2 = np.rot90(dx2)
    dZ_dy2 = convolve(Z,-dy2)

    Az = np.degrees(np.arctan2(dZ_dx2,dZ_dy2))
    return Az