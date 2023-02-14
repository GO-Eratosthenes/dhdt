import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .image_io import output_image
from ..generic.mapping_tools import covar2err_ellipse

def plot_displacement_vectors(X, Y, U, V, U_err, V_err, ρ_err, ax=None,
                              scaling=(10, 100), sampling=10, dots=True,
                              M=None):
    """
    plot displacement vectors with error ellipses

    Parameters
    ----------
    X,Y : numpy.ndarray, size=(m,n)
        grid with horizontal and vertical coordinates
    U,V : numpy.ndarray, size=(m,n)
        grid with horizontal and vertical displacements
    U_err,V_err : numpy.ndarray, size=(m,n)
        grid with horizontal and vertical precision estimate
    ρ_err : numpy.ndarray, size=(m,n)
        grid with orientation of error ellipse
    scaling : list of floats, default=(10.,100.)
        scaling of the displacement vectors and the error ellipses
    sampling : integer, default=10
        the interval of sampling over the grid
    dots : boolean, default=True
        plot starting position of the vector
    M : numpy.ndarray, size=(m,n), dtype=bool, default=None
        mask specifying which data in the grid to exclude, hence True specify
        data not to include
    """
    if M is None:
        M = np.zeros_like(X, dtype=bool)
    if ax is None:
        fig, ax = plt.subplots()

    grd_1,grd_2 = np.zeros_like(X), np.zeros_like(X)
    grd_1[::sampling,...], grd_2[...,::sampling] = 1, 1
    sparse_grd = np.logical_and(grd_1, grd_2)

    idxs = np.where(np.logical_and(sparse_grd, np.invert(M)))

    if dots:
        plt.scatter(X[idxs],Y[idxs],s=2,c='k', alpha=.5)

    for cnt in range(idxs[0].size):
        # get displacement vectors
        X_beg = X[idxs[0][cnt],idxs[1][cnt]]
        X_end = X[idxs[0][cnt],idxs[1][cnt]] + \
                np.multiply(U[idxs[0][cnt],idxs[1][cnt]], scaling[0])
        Y_beg = Y[idxs[0][cnt],idxs[1][cnt]]
        Y_end = Y[idxs[0][cnt],idxs[1][cnt]] + \
                np.multiply(V[idxs[0][cnt],idxs[1][cnt]], scaling[0])
        plt.plot(np.stack((X_beg, X_end)), np.stack((Y_beg, Y_end)),
                 '-k', alpha=.5)

        # orientation of the ellipse
        lam_1,lam_2,θ = covar2err_ellipse(U_err[idxs[0][cnt],idxs[1][cnt]],
                                              V_err[idxs[0][cnt],idxs[1][cnt]],
                                              ρ_err[idxs[0][cnt],idxs[1][cnt]]
                                              )
        ell = Ellipse((X_end, Y_end), width=lam_1*scaling[1],
                      height=lam_2*scaling[1], angle=θ,
                      facecolor='none', edgecolor='black')
        ax.add_patch(ell)
    return ax

# plot_strain_rate_crosses
# ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
# ...             arrowprops=dict(arrowstyle="->", edgecolor='red'))

def disco_displacement(V_abs, fpath, fname, Msk=None,
                       cmap='twilight',counts=32, factor=10):
    """ create a velocity sequence in disco style

    Parameters
    ----------
    V_abs : numpy.ndarray, size=(m,n), dtype={integer,float}
        array with speed, that is absolute velocity
    fpath : string
        direction to put imagery data
    fname : string
        naming of the files
    Msk : numpy.ndarray, size=(m,n), dtype={integer,float}
        array with locations with correct entities in V_abs
    cmap : string
        specifying a colormap
    counts : integer, {x ∈ ℕ}
        amount of frames to produce
    factor : float, {x ∈ ℝ | x ≥ 0}
        speed of the propagation

    Examples
    --------
    >>> import os
    >>> import numpy as np

    >>> x,y = np.linspace(-1, 1, 101), np.linspace(-1, 1, 101)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = 4 - np.abs((X*Y + Y**3) - X**2 - X)
    >>> M = np.hypot(X,Y)<1

    >>> fname = 'test'
    >>> fpath = os.path.join(os.getcwd(), 'disco')
    >>> disco_displacement(Z, fpath, fname, Msk=M)

    Look in the current working folder, there a folder called "disco" with a
    collection of images. These can be merged together in the terminal to an
    animated GIF. This is done via ImageMagick, through the following command:

    >>> convert -delay 5 -loop 0 *.jpg dummy.gif
    >>> convert dummy.gif -strip -coalesce -layers Optimize test.gif
    >>> rm dummy.gif

    A movie can be created as well, via ffmpeg:

    >>> ffmpeg ffmpeg -r 10 -i test%03d.jpg -vcodec mpeg4 -y test.mp4

    or convert the movie back to an animated GIF

    >>> ffmpeg -i test.mp4 -r 10 -vf scale=512:-1 test.gif
    """

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    v_mod = factor/np.max(V_abs)

    for count in range(counts):
        v_bias = (count/counts)*v_mod
        fringe = (np.divide(1, 1 + V_abs) + v_bias) % v_mod
        outputname = os.path.join(fpath,
                                  fname+str(count).zfill(3)+'.jpg')
        if Msk is not None:
            np.place(fringe, ~Msk, np.nan)

        output_image(fringe, outputname, cmap=cmap)
    return
