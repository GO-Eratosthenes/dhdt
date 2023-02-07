import numpy as np

from scipy import ndimage
from ..generic.handler_im import get_grad_filters

def estimate_strain_rates(X,Y,U,V):
    dX,dY = get_grad_filters(ftype='kroon', tsize=3, order=1, indexing='xy')
    #todo scale to size of the grid spacing

    dUdX, dVdY = ndimage.convolve(U, dX), ndimage.convolve(V, dY)
    dVdX, dUdY = ndimage.convolve(V, dX), ndimage.convolve(U, dY)
    E_xy = .5*dVdX + .5*dUdY
    E_xx, E_yy = dUdX, dVdY
    return E_xx, E_yy, E_xy

def estimate_principle_strain(E_xx, E_yy, E_xy):
    E_min, E_plu = E_xx - E_yy, E_xx + E_yy

    lamb = np.sqrt(np.divide(E_min,2)**2 + E_xy**2)
    E_max, E_min = E_plu/2 + lamb, E_plu/2 - lamb
    Theta = np.rad2deg(np.arctan2(2*E_xy, E_min) / 2)
    return Theta, E_max, E_min
