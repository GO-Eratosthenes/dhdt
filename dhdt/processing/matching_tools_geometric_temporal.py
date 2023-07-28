from scipy.signal import convolve2d


def spatial_temporal_evolution(C_12, C_23):
    r""" convolve displacement estimates in the spatial domain, see [AK17]_
    
    Parameters
    ----------
    C_12 : numpy.array, size=(m,n), ndim={2,3}
        similarity function of first interval.
    C_23 : numpy.array, size=(m,n), ndim={2,3}
        similarity function of second interval.

    Returns
    -------
    C_13 : numpy.array, size=(m,n), ndim={2,3}
        similarity function of total interval.

    See Also
    --------
    spatial_temporal_evolution
    
    Notes
    ----- 
    The matching equations are as follows:

    .. math:: \mathbf{C}_{12} = \mathbf{I}_1 \circledast \mathbf{1}_{2}
    .. math:: \mathbf{C}_{13} = \mathbf{C}_{12} \circledast \mathbf{C}_{23}

    Schematically this looks like:

        .. code-block:: text

          ┌-------D13---------┐
          |                   |
          ┴---D12---┴---D23---┴
         ------------------------------> time

    References
    ----------
    .. [AK17] Altena & Kääb. "Elevation change and improved velocity retrieval
              using orthorectified optical satellite data from different orbits"
              Remote sensing vol.9(3) pp.300, 2017.
    """
    C_13 = convolve2d(C_12, C_23, mode='same')
    return C_13


def frequency_temporal_evolution(Q_12, Q_23):
    r""" convolve displacement estimates in the frequency domain, see [AK17]_.
    
    Parameters
    ----------
    Q_12 : numpy.array, size=(m,n), ndim={2,3}
        cross-power spectrum of first interval.
    Q_23 : numpy.array, size=(m,n), ndim={2,3}
        cross-power spectrum of second interval.

    Returns
    -------
    Q_13 : numpy.array, size=(m,n), ndim={2,3}
        cross-power spectrum of total interval.

    See Also
    --------
    spatial_temporal_evolution
    
    Notes
    ----- 
    The matching equations are as follows:

    .. math:: \mathbf{S}_1, \mathbf{S}_2 = \mathcal{F}[\mathbf{I}_1], \mathcal{F}[\mathbf{I}_2]
    .. math:: \mathbf{Q}_{12} = \mathbf{S}_1 \odot \mathbf{S}_{2}^{\star}
    .. math:: \mathbf{Q}_{13} = \mathbf{Q}_{12} \odot \mathbf{Q}_{23}

    See also Equation 17 in [1].

    Schematically this looks like:

        .. code-block:: text

          ┌-------D13---------┐
          |                   |
          ┴---D12---┴---D23---┴
         ------------------------------> time

    References
    ----------
    .. [AK17] Altena & Kääb. "Elevation change and improved velocity retrieval
              using orthorectified optical satellite data from different orbits"
              Remote sensing vol.9(3) pp.300, 2017.
    """
    Q_13 = Q_12 * Q_23
    return Q_13


# def find_peak_along_flightline(C,e):
