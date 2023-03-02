import random

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import siegelslopes
from sklearn.decomposition import fastica

from dhdt.generic.unit_check import are_three_arrays_equal
from dhdt.processing.gis_tools import get_intersection


def bispectral_minimum(B1, B2, tiks=256, qntl=0.1, im_show=False):
    """ find the dark pixel via intersecting the envelop of a bispectral plot.
    Method is based on [Cr87]_, though homogenous regions were selected by hand.
    Thus here a data driven approach is used, in line with [Ka14]_.

    Parameters
    ----------
    B1,B2 : numpy.ndarray, size=(m,n)
        grid with intensities with a certain spectral band
    tiks : integer
        amount of bins
    qntl : float, range = 0...1
        quantile to be used to select the outerbounds

    Returns
    -------
    x_ij :

    References
    ----------
    .. [Cr87] Crippen, "The regression intersection method of adjusting image
              data for band ratioing" International journal of remote sensing,
              vol.8 pp.137-155, 1987.
    .. [Ka14] Kao et al. "Calibrated ratio approach for vegetation detection in
              shaded areas" Journal of applied remote sensing, vol.8,
              pp.083543-1-19, 2014
    """
    H, _, _ = np.histogram2d(B1.ravel(), B2.ravel(),
                             bins=tiks, range=[[0, tiks], [0, tiks]])
    H_0, H_1 = np.cumsum(H, axis=0), np.cumsum(H, axis=1)
    H_0 = np.divide(H_0, np.tile(H_0[-1, :], (H.shape[1], 1)), where=H_0 != 0)
    H_1 = np.divide(H_1, np.tile(H_1[:, -1], (H.shape[0], 1)).T, where=H_1 != 0)

    H_0[H_0 > qntl] = 0
    i_0 = np.argmax(H_0, axis=0).astype(float)
    IN = i_0 != 0
    i_0, j_0 = i_0[IN], np.linspace(0, tiks - 1, tiks)[IN]

    H_1[H_1 > qntl] = 0
    j_1 = np.argmax(H_1, axis=1).astype(float)
    IN = j_1 != 0
    i_1, j_1 = np.linspace(0, tiks - 1, tiks)[IN], j_1[IN]

    res_0, res_1 = siegelslopes(i_0, j_0), siegelslopes(i_1, j_1)

    ij_0 = np.array([[j_0[0], res_0[1] + res_0[0]*j_0[0]],
                    [j_0[-1], res_0[1] + res_0[0]*j_0[-1]]])
    ij_1 = np.array([[j_1[0], res_1[1] + res_1[0]*j_1[0]],
                    [j_1[-1], res_1[1] + res_1[0]*j_1[-1]]])

    x_ij = get_intersection(ij_0[0,:], ij_0[1,:], ij_1[0,:], ij_1[1,:])

    if im_show:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(H, cmap=plt.cm.gray)
        ax.scatter(j_0,i_0,color='r', marker='.')
        ax.scatter(j_1,i_1,color='b', marker='.')
        ax.plot(ij_0[:,0], ij_0[:,1], 'r-')
        ax.plot(ij_1[:,0], ij_1[:,1], 'b-')
        plt.gca().invert_yaxis()
        plt.xlim([0, tiks]), plt.ylim([0, tiks])
        plt.show()

    return x_ij

def principle_component_analysis(X):
    """ principle component analysis (PCA)

    Parameters
    ----------
    X : numpy.ndarray, size=(m,b)
        array of the band image

    Returns
    -------
    eigen_vecs : numpy.ndarray, size=(b,b)
        array with eigenvectors
    eigen_vals : numpy.ndarray, size=(b,1)
        vector with eigenvalues

    """
    assert type(X)==np.ndarray, ('please provide an array')

    # Data matrix X, assumes 0-centered
    n, m = X.shape
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n - 1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    return eigen_vecs, eigen_vals


def pca_rgb_preparation(Blue, Green, Red, min_samp=1e4):
    are_three_arrays_equal(Blue,Green,Red)

    OK = Blue != 0  # boolean array with data

    r,g,b = np.mean(Red[OK]), np.mean(Green[OK]), np.mean(Blue[OK])
    # shift axis origin
    Red,Green,Blue = Red-r, Green-g, Blue-b
    del r, g, b

    # sampleset
    Nsamp = int(
        min(min_samp, np.sum(OK)))  # try to have a sampleset of 10'000 points
    sampSet, other = np.where(OK)  # OK.nonzero()
    sampling = random.choices(sampSet, k=Nsamp)
    del sampSet, Nsamp

    Red = Red.ravel(order='F')
    Green = Green.ravel(order='F')
    Blue = Blue.ravel(order='F')

    X = np.transpose(np.array(
        [Red[sampling], Green[sampling], Blue[sampling]]))
    return X


def independent_component_analysis(Blue, Green, Red, Near, min_samp=1e4):
    OK = Blue != 0
    # sampleset
    Nsamp = int(
        min(min_samp, np.sum(OK)))  # try to have a sampleset of 10'000 points
    sampSet, other = np.where(OK)  # OK.nonzero()
    sampling = random.choices(sampSet, k=Nsamp)
    del sampSet, Nsamp
    X = np.array([Blue.flatten()[sampling],
                  Green.flatten()[sampling],
                  Red.flatten()[sampling],
                  Near.flatten()[sampling]]).T
    K,W,S,X_mean = fastica(X, return_X_mean=True)

    # reconstruct components
    first = W[0,0]*(Blue-X_mean[0]) +W[0,1]*(Green-X_mean[1]) \
            +W[0,2]*(Red-X_mean[2]) +W[0,3]*(Near-X_mean[3])
    secnd = W[1,0]*(Blue-X_mean[0]) +W[1,1]*(Green-X_mean[1]) \
            +W[1,2]*(Red-X_mean[2]) +W[1,3]*(Near-X_mean[3])
    third = W[2,0]*(Blue-X_mean[0]) +W[2,1]*(Green-X_mean[1]) \
            +W[2,2]*(Red-X_mean[2]) +W[2,3]*(Near-X_mean[3])
    forth = W[3,0]*(Blue-X_mean[0]) +W[3,1]*(Green-X_mean[1]) \
            +W[3,2]*(Red-X_mean[2])+W[3,3]*(Near-X_mean[3])
    return first, secnd, third, forth