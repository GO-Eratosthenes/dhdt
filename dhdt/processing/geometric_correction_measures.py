import numpy as np

from scipy.stats import spearmanr, wilcoxon, ttest_1samp

def bad_point_propagation(dXY, r=1.):
    """ see [Go09]_

    Parameters
    ----------
    dXY : numpy.ndarray, size=(k,2)
        planimetric corrections / misfits
    r : float
        threshold radius of the ellipse

    Returns
    -------
    BPP : float, range=0...1
        proportion of outliers

    References
    ----------
    .. [Go09] Gonçalves et al., "Measures for an objective evaluation of the
              geometric correction process quality", IEEE geoscience and remote
              sensing letters, vol.6(2) pp.292-296, 2009.
    """
    n = dXY.shape[1]
    C = np.cov(dXY, rowvar=False)
    C *= r
    Cinv = np.linalg.inv(C)

    # calculate in normalized ellipse frame
    D = np.einsum('ij,ij->i',np.einsum('kj,jl->kl', dXY,Cinv),dXY)
    # point inside error ellipse
    BPP = np.divide(np.sum(D<1), n)
    return BPP

def skew_distribution(dXY):
    """ skeweness between axis, following [Go09]_.

    Parameters
    ----------
    dXY : numpy.ndarray, size=(k,2)
        planimetric corrections / misfits

    Returns
    -------
    Skew : float, range=0...1
        amount of correlation between both axes

    References
    ----------
    .. [Go09] Gonçalves et al., "Measures for an objective evaluation of the
              geometric correction process quality", IEEE geoscience and remote
              sensing letters, vol.6(2) pp.292-296, 2009.
    """
    n = dXY.shape[1]
    if n<20: # Spearman
        Skew = spearmanr(dXY[:,0], dXY[:,1])[0]
    else: # Pearson
        Skew = np.corrcoef(dXY[:,0], dXY[:,1])
    return Skew

def scat_distribution(XY, XY_dim):
    """ look at the distribution of the control points, following [1].

    Parameters
    ----------
    XY : numpy.ndarray, size=(k,2)
        location of the control points
    XY_dim : list, size=2
        dimension of the domain

    Returns
    -------
    Scat : float, range=0...1
        scattering distibution
    """
    r = .5*np.mean(XY_dim)

    # compute distances
    xy = XY[:,0] + 1j*XY[:,1]
    xy_1,xy_2 = np.meshgrid(xy, xy)
    dxy = np.abs(xy_1-xy_2)

    # look at statistics of the neighborhood
    n = XY.shape[1]
    D = np.eye(n).astype(bool)
    dxy[D] = np.nan
    dist = np.nanmean(dxy, axis=1)

    if n < 30:  # Wilcoxon
        res = wilcoxon(dist)
        Scat = 1 - res.pvalue
    else:
        res = ttest_1samp(dist, popmean=r)
        Scat = 1 - res.pvalue
    return Scat
