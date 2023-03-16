import numpy as np
from scipy import ndimage
from skimage.morphology import dilation, disk

from .velocity_tools import make_seeds

def directional_line(tsize,θ):
    if isinstance(tsize, int):
        grd = np.meshgrid(np.linspace(-tsize/2,+tsize/2,tsize),
                          np.linspace(-tsize/2,+tsize/2,tsize))
    else:
        grd = np.meshgrid(np.linspace(-tsize[0] / 2, +tsize[0] / 2, tsize[0]),
                          np.linspace(-tsize[1] / 2, +tsize[1] / 2, tsize[1]))
    grd_th = np.sin(np.rad2deg(θ))*grd[1] + \
             np.cos(np.rad2deg(θ))*grd[0]
    L = np.exp(-np.abs(grd_th))
    return L

def directional_spread(buffer):
    m = buffer.size // 2
    d = int(np.sqrt(buffer.size))

    seeds = np.round(buffer)
    thetas = buffer-seeds
    θ = thetas[m]

    central_pix = np.sum(np.multiply(seeds,
                                     directional_line(d,θ).flatten()))
    return central_pix

def crevasse_cartography(θ,Score=None,scaling=1): #todo: LIC
    S = np.zeros_like(θ, dtype=float)
    i,j = make_seeds(S, n=np.ceil(S.size/100))
    S[i.astype(int), j.astype(int)] = 1.

    S = dilation(S, footprint=disk(2))

    # doing a trick
    S += np.divide(θ,360)

    I_new = ndimage.generic_filter(S, directional_spread,
                                   size=(13,13))

    C = None
    return C


