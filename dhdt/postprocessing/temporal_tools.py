import numpy as np

from dhdt.preprocessing.image_transforms import mat_to_gray

def get_temporal_incidence_matrix(T_1, T_2, T_step, open_network=True):
    """

    Parameters
    ----------
    T_1 : numpy.ndarray, size=(m,), dtype=datetime64
        first temporal instance or start of the composed dataset
    T_2 : numpy.ndarray, size=(m,), dtype=datetime64
        last temporal instance or end of the composed dataset
    T_step : numpy.ndarray, size=(n,), dtype=datetime64
        temporal coverage and resolution of the preposed product
    open_network : bool, default=True
        use instances that also overarch the time period, see the difference
        between an 'open' and 'closed' network in [Al19]_.

    Returns
    -------
    A : numpy.ndarray, size=(m,n)
        incidence matrix, see literature [SB97]_ and [Al19]_ for implementation
        details. An assessment on this matrix framework is provided in [Ch22]_

    References
    ----------
    .. [SB97] Strang & Borre, "Linear algebra, geodesy and GPS",
              Wellesley-Cambridge Press, 1997.
    .. [Al19] Altena et al., "Extracting recent short-term glacier velocity
              evolution over southern Alaska and the Yukon from a large
              collection of Landsat data", The cryosphere, vol.13 pp.795–814,
              2019.
    .. [Ch22] Charrier et al. "Fusion of multi-temporal multi-sensor velocities
              using temporal closure of fractions of displacements", IEEE
              geoscience and remote sensing letters, vol.19 pp.1-5, 2022.
    """
    m, n = T_1.size, T_step.size-1

    # convert to integer values, at daily resolution
    T_1 = T_1.astype('datetime64[D]').astype(int)
    T_2 = T_2.astype('datetime64[D]').astype(int)
    T_int = T_step.astype('datetime64[D]').astype(int)
    T_start, T_end = T_int[:-1], T_int[1:]
    dT_step = np.diff(T_int)

    A_start, A_end = np.tile(T_start, (m,1)), np.tile(T_end, (m,1))
    A_step = np.tile(dT_step, (m,1))
    dT_start = np.tile(np.atleast_2d(T_2).T, (1,n)) - A_start
    dT_end = A_end - np.tile(np.atleast_2d(T_1).T, (1, n))

    OUT = np.logical_or(np.sign(dT_start) == -1,
                        np.sign(dT_end) == -1)
    np.putmask(dT_start, OUT, 0.)
    np.putmask(dT_end, OUT, 0.)

    dT_start = np.minimum(dT_start, A_step)
    dT_end = np.minimum(dT_end, A_step)

    if open_network:
        dT_end = A_step - dT_end
        np.putmask(dT_end, dT_end==A_step, 0)
        A = np.divide(dT_start-dT_end, A_step)
    else:
        IN = np.logical_and(dT_start==A_step, dT_end==A_step)
        A = IN.astype(float)
    return A

def get_temporal_weighting(T_1, T_2, covariance=False, corr_coef=.5):
    """

    Parameters
    ----------
    T_1 : numpy.ndarray, size=(m,), dtype=datetime64
        first temporal instance or start of the composed dataset
    T_2 : numpy.ndarray, size=(m,), dtype=datetime64
        last temporal instance or end of the composed dataset
    covariance : bool, default=True
        co-variances are given when at least one of the instances have the same
        date
    corr_coef : float
        the correlation coefficient to place in the off-diagonal elements

    Returns
    -------
    W : numpy.ndarray, size=(m,m)
        weighting matrix, diagonal following [Mo17]_

    References
    ----------
    .. [Mo17] Mouginot et al., "Comprehensive annual ice sheet velocity
              mapping using Landsat-8, Sentinel-1, and RADARSAT-2 data", Remote
              sensing, vol.9(4) pp.364, 2017.
    """
    # convert to integer values, at daily resolution
    T_1 = T_1.astype('datetime64[D]').astype(int)
    T_2 = T_2.astype('datetime64[D]').astype(int)

    m = T_1.size
    W = np.zeros((m,m), dtype=float)
    dT = T_2 - T_1
    dT = mat_to_gray(dT, vmin=0.1) # normalize weights to 0.1 ... 1.0
    np.fill_diagonal(W, dT)

    if covariance:
        instances = np.unique(np.stack((T_1, T_2)))
        for instance in instances:
            i, j = np.where(T_1==instance)[0], np.where(T_2==instance)[0]
            ij = np.concatenate([np.atleast_1d(i),np.atleast_1d(j)])
            if ij.size>1:
                I,J = np.meshgrid(ij,ij)
                I,J = I.flatten(), J.flatten()
                for c in range(I.size):
                    if I[c]!=J[c]: W[I[c],J[c]] = corr_coef
    return W