import numpy as np

from scipy.stats import spearmanr, wilcoxon, ttest_1samp
from dhdt.generic.unit_conversion import deg2compass

def doubling_the_angles(θ):
    ψ = np.mod(2*deg2compass(θ),360)
    return ψ

def _nchoosek(n, k):
    if k == 0:
        r = 1
    else:
        r = n/k * _nchoosek(n-1, k-1)
    return round(r)

def ajne_test(dXY, α=0.05):
    """ implementing the Hodges-Ajne test, see also [BP97]_ for implementation
    details. Hence, the direction of the misfits are uniformly distributed.

    Parameters
    ----------
    dXY : numpy.ndarray, size=(k,2)
        planimetric corrections / misfits
    α : float, range=0...1
        significance level

    References
    ----------
    .. [BP97] Buiten & van Putten, "Quality assessment of remote sensing image
              registration—analysis and testing of control point residuals"
              ISPRS journal of photogrammetry and remote sensing vol.52(2)
              pp.57-73, 1997.
    .. [Aj68] Ajne, "A simple test for uniformity of a circular distribution",
              Biometrika vol.55(2) pp.343-354, 1968.
    .. [Ho55] Hodges, "A bivariate sign test." The annals of mathematical
              statistics vol.26(3) pp.523-527, 1955.
    """
    ψ = doubling_the_angles(np.rad2deg(np.angle(dXY[:,0] + 1j*dXY[:,0])))
    n = ψ.size

    deg = np.linspace(0,359,360)
    m1 = np.zeros((n))
    for i, θ in enumerate(deg):
        m1[i, ...] = np.sum(((α > θ) & (α < 360 + α)))
    m2 = n - m1
    m = np.stack((m1,m2)).ravel().min()
    if n<50: # use [Aj68]_
        A = np.pi*np.sqrt(n) / 2 / (n - 2*m)
        p_val = np.divide(np.sqrt(2 * np.pi),
                          A * np.exp(-np.pi**2 / 8 / A**2))
    else: # use [Ho55]_
        p_val = 2**(1-n) * (n-2*m) * _nchoosek(n,m)
    return p_val

def moore_test(dXY):
    cXY = dXY[:,0] + 1j*dXY[:,1]
    l, ψ = np.abs(cXY), np.angle(cXY)
    t = np.argsort(l)

    n = ψ.size

    C, S = np.sum(t*np.cos(ψ)), np.sum(t*np.sin(ψ))
    D = np.sqrt(C**2 + S**2)
    D_star = np.divide(D, np.power(n, 3/2))

    return D_star

def spearman_test(dXY):
    """ implementing the Spearman test, see also [BP97]_ for implementation
    details.

    Parameters
    ----------
    dXY : numpy.ndarray, size=(k,2)
        planimetric corrections / misfits

    References
    ----------
    .. [BP97] Buiten & van Putten, "Quality assessment of remote sensing image
              registration—analysis and testing of control point residuals"
              ISPRS journal of photogrammetry and remote sensing vol.52(2)
              pp.57-73, 1997.

    """
    r,s = np.argsort(dXY[:,0]), np.argsort(dXY[:,1])
    n = dXY.shape[0]
    r_2 = np.divide(1 - 6*np.sum((r-s)**2),
                    n*(n**2 - 1))
    # student t-test

    return verdict, p_value

def rayleigh_test(dXY):
    """ see [Ba81]_

    Parameters
    ----------
    dXY : numpy.ndarray, size=(k,2)
        planimetric corrections / misfits

    Returns
    -------
    z : float
        test statistic

    References
    ----------
    .. [Ba81] Batschelet, "Circular statistics in biology." Academic press,
              1981.
    """
    n = dXY.shape[0]
    r = np.mean(dXY, axis=0) # (1.3.6)  in [Ba81]_

    if n>30:
        z = n*np.power(r,2) # (1.3.6)  in [Ba81]_
    else:
        z = r
    return z

def v_test(dXY, e):
    """ see [Ba81]_

    Parameters
    ----------
    dXY : numpy.ndarray, size=(k,2)
        planimetric corrections / misfits
    e : numpy.ndarray, size=(2,)
        preferred direction, in the form of a unit vector

    Returns
    -------
    u : float
        test statistic

    References
    ----------
    .. [Ba81] Batschelet, "Circular statistics in biology." Academic press,
              1981.
    """
    n = dXY.shape[0]

    ϕ = np.mean(dXY, axis=0)                        # (1.3.6)  in [Ba81]_
    r = np.linalg.norm(ϕ)
    ϕ_e = np.divide(ϕ, np.linalg.norm(ϕ))
    e = np.divide(e, np.linalg.norm(e))
    v = r*np.cos(np.clip(np.dot(ϕ_e, e), -1.0, 1.0))# (4.3.1)  in [Ba81]_
    u = np.sqrt(2*n)*v                              # (4.3.2)  in [Ba81]_
    return u

def rao_test(dXY):
    """ see [Ba81]_

    Parameters
    ----------
    dXY : numpy.ndarray, size=(k,2)
        planimetric corrections / misfits

    Returns
    -------
    U : float
        test statistic

    References
    ----------
    .. [Ba81] Batschelet, "Circular statistics in biology." Academic press,
              1981.
    """
    n = dXY.shape[0]

    ψ = np.rad2deg(np.angle(dXY[:,0] + 1j*dXY[:,1]))
    ψ = np.argsort(deg2compass(ψ))

    T = np.zeros_like(ψ)
    T[:-1] = ψ[:-1] - ψ[1:]
    T[-1] = 360 + ψ[0] - ψ[-1]                  # (4.6.1)  in [Ba81]_

    dT = 360/n
    U = .5 * np.sum(np.abs(T-dT))               # (4.6.4)  in [Ba81]_
    return U

def kuiper_test(dXY):
    """ see [Ba81]_

    Parameters
    ----------
    dXY : numpy.ndarray, size=(k,2)
        planimetric corrections / misfits

    Returns
    -------
    U : float
        test statistic

    References
    ----------
    .. [Ba81] Batschelet, "Circular statistics in biology." Academic press,
              1981.
    """
    n = dXY.shape[0]
    ψ = np.rad2deg(np.angle(dXY[:,0] + 1j*dXY[:,1]))
    ψ = np.argsort(deg2compass(ψ))

    cdf = np.linspace(0,1,n)
    D = ψ - cdf
    Dmin, Dplus = np.min(D), np.max(D)
    V_n = Dplus + Dmin                          # (4.9.1)  in [Ba81]_
    K = np.sqrt(n)*V_n                          # (4.9.2)  in [Ba81]_
    return K

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
    """ look at the distribution of the control points, following [Go09]_.

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

    References
    ----------
    .. [Go09] Gonçalves et al., "Measures for an objective evaluation of the
              geometric correction process quality", IEEE geoscience and remote
              sensing letters, vol.6(2) pp.292-296, 2009.
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
