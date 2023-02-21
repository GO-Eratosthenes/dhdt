import numpy as np

from ..generic.filtering_statistical import mad_filtering

from .matching_tools_frequency_filters import normalize_power_spectrum, \
    construct_phase_values, cross_spectrum_to_coordinate_list

def list_phase_metrics():
    """ list the abbreviations of the different implemented phase metrics,
    there are:
        * :func:`'phase_fit' <phase_fitness>` : the absolute score
        * :func:`'phase_sup' <phase_support>` : the primary peak ratio i.r.t. the second peak

    See Also
    --------
    get_phase_metric
    """
    metrics_list = ['phase_fit', 'phase_sup']
    return metrics_list

def get_phase_metric(Q, di, dj, metric='phase_fit'):
    """ redistribution function, to get a metric from a matching score surface

    Parameters
    ----------
    Q : numpy.array, size=(m,n), type=complex
        phase angles of the cross-correlation spectra
    di,dj : {float,integer}
        estimated displacement
    metric : string
        abbreviation for the metric type to be calculated, for the options see
        :func:`list_pahse_metrics` for the options

    Returns
    -------
    score : float
        metric of the matching phase angle in relation to its surface

    See Also
    --------
    list_phase_metrics
    dhdt.processing.matching_tools_spatial_metrics.list_matching_metrics

    """
    # admin
    assert type(Q) == np.ndarray, ('please provide an array')
    if Q.size==0: return None

    # redistribute correlation surface to the different functions
    if metric in ['phase_fit', 'phase_fitness', 'snr']:
        score = phase_fitness(Q, di, dj)
    elif metric in ['phase_sup', 'phase_support']:
        score = phase_support(Q, di, dj)
    else:
        score = None
    return score

def phase_fitness(Q, di, dj, norm=2):
    """ estimate the goodness of fit of the phase diffence

    Parameters
    ----------
    Q : numpy.array, size={(m,n), (k,3)}
        cross-spectrum
    di,dj : {float,integer}
        estimated displacement
    norm : integer
        norm for the difference

    Returns
    -------
    fitness : float, range=0...1
        goodness of fit, see also [1]

    See Also
    --------
    signal_to_noise, phase_support

    References
    ----------
    .. [1] Leprince, et.al. "Automatic and precise orthorectification,
       coregistration, and subpixel correlation of satellite images,
       application to ground deformation measurements", IEEE Transactions on
       geoscience and remote sensing vol. 45.6 pp. 1529-1558, 2007.
    """
    assert type(Q) == np.ndarray, ('please provide an array')
    # admin
    data = cross_spectrum_to_coordinate_list(Q)
    C = construct_phase_values(data[:,0:2], di, dj)

    # calculation
    QC = (data[:,-1] - C) ** norm
    dXY = np.abs(QC)
    fitness = (1-np.divide(np.sum(dXY) , 2*norm*dXY.size))**norm
    return fitness

def phase_support(Q, di, dj, thres=1.4826):
    """ estimate the support of the fit for the phase differences

    Parameters
    ----------
    Q : numpy.array, size={(m,n), (k,3)}
        cross-spectrum
    di,dj : {float,integer}
        estimated displacement
    norm : integer
        norm for the difference

    Returns
    -------
    support : float, range=0...1
        ammount of support, see also [1]

    See Also
    --------
    phase_fitness

    References
    ----------
    .. [1] Altena & Leinss, "Improved surface displacement estimation through
       stacking cross-correlation spectra from multi-channel imagery",
       Science of Remote Sensing, vol.6 pp.100070, 2022.
    """
    assert type(Q) == np.ndarray, ('please provide an array')

    data = cross_spectrum_to_coordinate_list(Q)
    C = construct_phase_values(data[:,0:2], di, dj)

    # calculation
    QC = (data[:,-1] - C) ** 2
    dXY = np.abs(QC)
    IN = mad_filtering(dXY, thres=thres)

    support = np.divide(np.sum(IN), np.prod(IN.shape))
    return support

def signal_to_noise(Q, C, norm=2):
    """ calculate the signal to noise from a theoretical and an experimental
    cross-spectrum

    Parameters
    ----------
    Q : numpy.array, size=(m,n), dtype=complex
        cross-spectrum
    C : numpy.array, size=(m,n), dtype=complex
        phase plane
    norm : integer
        norm for the difference

    Returns
    -------
    snr : float, range=0...1
        signal to noise ratio, see also [1]

    See Also
    --------
    phase_fitness

    References
    ----------
    .. [1] Leprince, et.al. "Automatic and precise orthorectification,
       coregistration, and subpixel correlation of satellite images,
       application to ground deformation measurements", IEEE Transactions on
       geoscience and remote sensing vol. 45.6 pp. 1529-1558, 2007.
    """
    assert type(Q) == np.ndarray, ('please provide an array')
    assert type(C) == np.ndarray, ('please provide an array')

    Qn = normalize_power_spectrum(Q)
    Q_diff = np.abs(Qn-C)**norm
    snr = 1 - (np.sum(Q_diff) / (2*norm*np.prod(C.shape)))
    return snr