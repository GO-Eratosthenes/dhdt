import numpy as np

from dhdt.generic.unit_conversion import deg2compass
from dhdt.processing.matching_tools_frequency_filters import gradient_fourier


def drift_profile(Z, geoTransform, θ):
    """

    Parameters
    ----------
    Z : numpy.array, size=(m,n), dtype=float, unit=meter
        grid with elevation values
    geoTransform : tuple, size={(1,6), (1,8)}
        georeference transform of the array 'Z'
    θ : float, unit=degrees, range=-180...+180
        dominant wind direction

    Returns
    -------
    Y : numpy.array, size=(m,n), dtype=float

    References
    ----------
    .. [Ta75] Tabler, "Predicting profiles of snowdrifts in topographic
              catchments" Proceedings of the 43rd annual western snow
              conference, pp.87-97, 1975.
    """

    θ = deg2compass(θ)
    # get scaling
    spac = np.hypot(geoTransform[1], geoTransform[2])

    dx_1, dx_2, dx_3, dx_4 = 45., 15./2, (15+30)/2, (30+45)/2

    dx_1, dx_2, dx_3, dx_4 = dx_1/spac, dx_2/spac, dx_3/spac, dx_4/spac
    Z_n = np.divide(Z, spac) # normalize from metric to pixel scale

    # steerable filters
    dZdi_1,dZdj_1 = gradient_fourier(np.divide(Z_n, dx_1))
    X_1 = []

    dZdi_2, dZdj_2 = gradient_fourier(np.divide(Z_n, dx_2), scaling=dx_2)

    # cut-off

    Y = X_1 # eq.(1) in [1]

    return Y