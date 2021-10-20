from scipy.signal import convolve2d

def spatial_temporal_evolution(C_12,C_23):
    """ convolve displacement estimates in the spatial domain
    
    Parameters
    ----------
    C_12 : np.array, size=(m,n), ndim={2,3}
        similarity function of first interval.
    C_23 : np.array, size=(m,n), ndim={2,3}
        similarity function of second interval.

    Returns
    -------
    C_13 : np.array, size=(m,n), ndim={2,3}
        similarity function of total interval.

    See Also
    --------
    spatial_temporal_evolution
    
    Notes
    ----- 
    The matching equations are as follows:

    .. math:: \mathbf{C}_{12} = \mathbf{I}_1 \circledast \mathbf{1}_{2}
    .. math:: \mathbf{C}_{13} = \mathbf{C}_{12} \circledast \mathbf{C}_{23}

    .. [1] Altena & Kääb. "Elevation change and improved velocity retrieval 
       using orthorectified optical satellite data from different orbits" 
       Remote sensing vol.9(3) pp.300 2017.
    """    
    #   -------D13---------
    #  |                   |
    #  |                   |
    #  +---D12---+---D23---+
    # ------------------------------> time
    
    C_13 = convolve2d(C_12,C_23, mode='same')
    return C_13

def frequency_temporal_evolution(Q_12,Q_23):
    """ convolve displacement estimates in the frequency domain
    
    Parameters
    ----------
    Q_12 : np.array, size=(m,n), ndim={2,3}
        cross-power spectrum of first interval.
    Q_23 : np.array, size=(m,n), ndim={2,3}
        cross-power spectrum of second interval.

    Returns
    -------
    Q_13 : np.array, size=(m,n), ndim={2,3}
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

    .. [1] Altena & Kääb. "Elevation change and improved velocity retrieval 
       using orthorectified optical satellite data from different orbits" 
       Remote sensing vol.9(3) pp.300 2017.
    """
    #   -------D13---------
    #  |                   |
    #  |                   |
    #  +---D12---+---D23---+
    # ------------------------------> time
    Q_13 = Q_12*Q_23
    return Q_13 

# def find_peak_along_flightline(C,e):