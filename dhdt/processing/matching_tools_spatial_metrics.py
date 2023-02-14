import numpy as np

from skimage.morphology import extrema

from ..generic.unit_check import are_two_arrays_equal
from ..preprocessing.image_transforms import high_pass_im

def list_matching_metrics():
    """ list the abbreviations of the different implemented correlation metrices
    there are:
        * 'peak_abs' : the absolute score
        * 'peak_ratio' : the primary peak ratio i.r.t. the second peak
        * 'peak_rms' : the peak ratio i.r.t. the root mean square error
        * 'peak_ener' : the peaks' energy
        * 'peak_noise' : the peak score i.r.t. to the noise level
        * 'peak_conf' : the peak confidence
        * 'peak_entr' : the peaks' entropy

    See Also
    --------
    get_correlation_metric, entropy_corr, peak_confidence, peak_to_noise,
    peak_corr_energy, peak_rms_ratio, num_of_peaks, peak_winner_margin,
    primary_peak_margin, primary_peak_ratio
    """
    metrics_list = ['peak_ratio', 'peak_rms', 'peak_ener', 'peak_nois',
                    'peak_conf', 'peak_entr', 'peak_abs', 'peak_marg',
                    'peak_win', 'peak_num']
    return metrics_list

def get_correlation_metric(C, metric='peak_abs'):
    """ redistribution function, to get a metric from a matching score surface

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n)
        grid with correlation scores
    metric : string
        abbreviation for the metric type to be calculated, for the options see
        "list_matching_metrics" for the options

    Returns
    -------
    score : float
        metric of the matching peak in relation to its surface

    See Also
    --------
    list_matching_metrics, entropy_corr, peak_confidence, peak_to_noise,
    peak_corr_energy, peak_rms_ratio, num_of_peaks, peak_winner_margin,
    primary_peak_margin, primary_peak_ratio
    """
    # admin
    assert type(C) == np.ndarray, ('please provide an array')
    if C.size==0: return None

    # redistribute correlation surface to the different functions
    if metric in ['peak_ratio']:
        score = primary_peak_ratio(C)
    elif metric in ['peak_rms']:
        score = peak_rms_ratio(C)
    elif metric in ['peak_ener']:
        score = peak_corr_energy(C)
    elif metric in ['peak_noise']:
        score = peak_to_noise(C)
    elif metric in ['peak_conf']:
        score = peak_confidence(C)
    elif metric in ['peak_entr']:
        score = entropy_corr(C)
    elif metric in ['peak_marg']:
        score = primary_peak_margin(C)
    elif metric in ['peak_win']:
        score = peak_winner_margin(C)
    elif metric in ['peak_num']:
        score = num_of_peaks(C)
    else: # 'peak_abs'
        score = np.argmax(C)
    return score

def primary_peak_ratio(C):
    """ metric for uniqueness of the correlation estimate
    also known as, "Cross correlation peak ratio"

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        correlation or similarity surface

    Returns
    --------
    ppr : float, range={0..1}
        peak ratio between primary and secondary peak

    See Also
    --------
    entropy_corr, peak_confidence, peak_to_noise, peak_corr_energy,
    peak_rms_ratio, num_of_peaks, peak_winner_margin, primary_peak_margin,

    References
    ----------
    .. [1] Keane & Adrian, "Optimization of particle image velocimeters. I.
       Double pulsed systems" Measurement science and technology. vol.1 pp.1202,
       1990.
    .. [2] Charonk & Vlachos "Estimation of uncertainty bounds for individual
       particle image velocimetry measurements from cross-correlation peak
       ratio" Measurement science and technology. vol.24 pp.065301, 2013.
    .. [3] Xue et al. "Particle image velocimetry correlation signal-to-noise
       ratio metrics and measurement uncertainty quantification" Measurement
       science and technology, vol.25 pp.115301, 2014.
    """
    from .matching_tools import get_peak_indices

    assert type(C) == np.ndarray, ('please provide an array')

    idx, val = get_peak_indices(C, num_estimates=2)
    ppr = np.divide(val[0], val[1],
                    out=np.ones(1), where=val[1]!=0)
    return ppr

def primary_peak_margin(C):
    """ metric for dominance of the correlation estimate in relation to other
    candidates

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        correlation or similarity surface

    Returns
    --------
    ppm : float
        margin between primary and secondary peak

    See Also
    --------
    entropy_corr, peak_confidence, peak_to_noise, peak_corr_energy,
    peak_rms_ratio, num_of_peaks, peak_winner_margin, primary_peak_ratio

    References
    ----------
    .. [1] Hu & Mordohai, "A quantitative evaluation of confidence measures for
       stereo vision" IEEE transactions on pattern analysis and machine
       intelligence, vol.34(11) pp.2121-2133, 2012.
    """
    from .matching_tools import get_peak_indices

    assert type(C) == np.ndarray, ('please provide an array')

    idx, val = get_peak_indices(C, num_estimates=2)
    ppm = val[0] - val[1]
    return ppm

def peak_winner_margin(C):
    """ metric for dominance of the correlation estimate in relation to other
    candidates and their surrounding scores

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        correlation or similarity surface

    Returns
    --------
    ppm : float
        margin between primary and secondary peak

    See Also
    --------
    entropy_corr, peak_confidence, peak_to_noise, peak_corr_energy,
    peak_rms_ratio, num_of_peaks, primary_peak_margin, primary_peak_ratio

    References
    ----------
    .. [1] Scharstein & Szeliski, "Stereo matching with nonlinear diffusion"
       International journal of computer vision, vol.28(2) pp.155-174, 1998.
    """
    assert type(C) == np.ndarray, ('please provide an array')

    pwm = primary_peak_margin(C)/np.sum(C)
    return pwm

def num_of_peaks(C, filtering=True):
    """ metric for the uniqueness of a match

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        correlation or similarity surface
    filtering : boolean
        apply low-pass filtering, otherwise noise is also seen as a peak,
        see [2] for more motivation

    Returns
    --------
    ppm : float
        margin between primary and secondary peak

    See Also
    --------
    entropy_corr, peak_confidence, peak_to_noise, peak_corr_energy,
    peak_rms_ratio, peak_winner_margin, primary_peak_margin, primary_peak_ratio

    References
    ----------
    .. [1] Lefebvre et al., "A colour correlation-based stereo matching using
       1D windows" IEEE conference on signal-image technologies and internet-
       based system, pp.702-710, 2007.
    .. [2] Hu & Mordohai, "A quantitative evaluation of confidence measures for
       stereo vision" IEEE transactions on pattern analysis and machine
       intelligence, vol.34(11) pp.2121-2133, 2012.
    """
    assert type(C) == np.ndarray, ('please provide an array')

    if filtering==True: # low pass filtering
        C -= high_pass_im(C, radius=3)

    peak_C = extrema.local_maxima(C)
    nop = np.sum(peak_C)
    return nop

def peak_rms_ratio(C):
    """ metric for uniqueness of the correlation estimate
    this is a similar metric as implemented in ROI_PAC [2]

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        correlation or similarity surface

    Returns
    --------
    prmsr : float
        peak to root mean square ratio

    See Also
    --------
    entropy_corr, peak_confidence, peak_to_noise, peak_corr_energy,
    num_of_peaks, peak_winner_margin, primary_peak_margin, primary_peak_ratio

    References
    ----------
    .. [1] Xue et al. "Particle image velocimetry correlation signal-to-noise
       ratio metrics and measurement uncertainty quantification" Measurement
       science and technology, vol.25 pp.115301, 2014.
    .. [2] Rosen et al. "Updated repeat orbit interferometry package released"
       EOS, vol.85(5) pp.47
    """
    assert type(C) == np.ndarray, ('please provide an array')

    max_corr = np.amax(C)
    hlf_corr = np.divide( max_corr, 2)
    noise = C<=hlf_corr
    C_rms = np.sqrt((1/np.sum(noise))  * np.sum(C[noise]**2))

    prmsr = np.divide( max_corr**2, C_rms)
    return prmsr

def peak_corr_energy(C):
    """ metric for uniqueness of the correlation estimate,
    also known as, "signal to noise"

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        correlation or similarity surface

    Returns
    --------
    prmsr : float
        peak to correlation energy

    See Also
    --------
    entropy_corr, peak_confidence, peak_to_noise, peak_rms_ratio, num_of_peaks,
    peak_winner_margin, primary_peak_margin, primary_peak_ratio

    References
    ----------
    .. [1] Xue et al. "Particle image velocimetry correlation signal-to-noise
       ratio metrics and measurement uncertainty quantification" Measurement
       science and technology, vol.25 pp.115301, 2014.
    """
    assert type(C) == np.ndarray, ('please provide an array')

    max_corr = np.amax(C)
    E_c = np.sum(np.abs(C.flatten())**2)
    pce = np.divide(max_corr**2, E_c)
    return pce

def peak_to_noise(C):
    """ metric for uniqueness of the correlation estimate

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        correlation or similarity surface

    Returns
    --------
    snr : float
        signal to noise ratio

    See Also
    --------
    entropy_corr, peak_confidence, peak_corr_energy, peak_rms_ratio,
    num_of_peaks, peak_winner_margin, primary_peak_margin, primary_peak_ratio

    References
    ----------
    .. [1] de Lange et al. "Improvement of satellite radar feature tracking for
       ice velocity derivation by spatial frequency filtering" IEEE transactions
       on geosciences and remote sensing, vol.45(7) pp.2309--2318.
    .. [2] Heid & Kääb "Evaluation of existing image matching methods for
       deriving glacier surface displacements globally from optical satellite
       imagery." Remote sensing of environment vol.118 pp.339-355, 2012.
    """
    assert type(C) == np.ndarray, ('please provide an array')

    C = C.flatten()
    max_idx, max_corr = np.argmax(C), np.amax(C)
    mean_corr = np.mean(np.concatenate((C[:max_idx],
                                        C[max_idx+1:])))
    mean_corr = np.abs(mean_corr)
    snr = np.divide(max_corr, mean_corr,
                   out=np.ones(1), where=mean_corr!=0)
    return snr

def peak_confidence(C, radius=1):
    """ metric for steepness of a correlation peak

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        correlation or similarity surface
    radius : integer, {x ∈ ℕ | x ≥ 1}, default=1
        how far away should the sampling set be taken

    Returns
    --------
    q : float
        confidence measure

    See Also
    --------
    entropy_corr, peak_to_noise, peak_corr_energy, peak_rms_ratio, num_of_peaks,
    peak_winner_margin, primary_peak_margin, primary_peak_ratio

    References
    ----------
    .. [1] Erten et al. "Glacier velocity monitoring by maximum likelihood
       texture tracking" IEEE transactions on geosciences and remote sensing,
       vol.47(2) pp.394--405, 2009.
    .. [2] Erten "Glacier velocity estimation by means of polarimetric
       similarity measure" IEEE transactions on geosciences and remote sensing,
       vol.51 pp.3319--3327, 2013.
    """
    from .matching_tools import get_template

    assert type(C) == np.ndarray, ('please provide an array')

    C_max = np.argmax(C)
    ij = np.unravel_index(C_max, C.shape, order='F')
    idx_2, idx_1 = ij[::-1]

    C_sub = get_template(C, idx_1, idx_2, radius).flatten()

    C_mean, C_min = np.nanmean(C_sub), np.argmin(C_sub)
    q = np.divide(C_max - C_mean, C_mean - C_min)
    return q

def entropy_corr(C):
    """ metric for uniqueness of the correlation estimate

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        correlation or similarity surface

    Returns
    --------
    disorder : float
        entropy of the correlation surface

    See Also
    --------
    peak_confidence, peak_to_noise, peak_corr_energy, peak_rms_ratio,
    num_of_peaks, peak_winner_margin, primary_peak_margin, primary_peak_ratio

    References
    ----------
    .. [1] Scharstein & Szeliski, "Stereo matching with nonlinear diffusion"
       International journal of computer vision, vol.28(2) pp.155-174, 1998.
    .. [2] Xue et al. "Particle image velocimetry correlation signal-to-noise
       ratio metrics and measurement uncertainty quantification" Measurement
       science and technology, vol.25 pp.115301, 2014.
    .. [3] Sturges, "The choice of a class interval". Journal of the american
       statistical association. vol.21(153) pp.65–66.
    """
    assert type(C) == np.ndarray, ('please provide an array')

    sturges = 1.6 * (np.log2(C.size) + 1) # [1] uses 30, but [2] is more adaptive
    values, base = np.histogram(C.flatten(),
                                bins=int(np.ceil(sturges)))

    # normalize
    p = np.divide(values, C.size)
    # entropy calculation
    disorder = - np.sum(np.multiply(p, np.log(p, out=np.zeros_like(p),
                                              where=p!=0)))
    return disorder

def hessian_spread(C, intI, intJ):
    """

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n), dtype=float
        array with correlation values
    intI : integer, {x ∈ ℕ | x ≥ 0}
        location in rows of highest value in search space
    intJ : integer, {x ∈ ℕ | x ≥ 0}
        location in collumns of highest value in search space

    Returns
    -------
    cov_ii : float, real, unit=pixel
        first diagonal element of the co-variance matrix
    cov_jj : float, real, unit=pixel
        second diagonal element of the co-variance matrix
    cov_ij : float, real, unit=pixel
        off-diagonal element of the co-variance matrix

    Notes
    -----
    The image frame is used in this function, hence the difference is:

        .. code-block:: text

          coordinate |              +--------> collumns
          system 'ij'|              |
                     |              |
                     |       j      |  image frame
             --------+-------->     |
                     |              |
                     |              v
                     | i            rows
                     v

    References
    ----------
    .. [1] Rosen et al. "Updated repeat orbit interferometry package released"
       EOS, vol.85(5) pp.47, 2004.
    .. [2] Casu et al. "Deformation time-series generation in areas
       characterized by large displacement dynamics: The SAR amplitude pixel-
       offset SBAS Technique" IEEE transactions in geoscience and remote
       sensing vol.49(7) pp.2752--2763, 2011.
    """
    assert type(C) == np.ndarray, ('please provide an array')

    if (intI==0) or (intI+1==C.shape[0]) or (intJ==0) or (intJ+1==C.shape[1]):
        C = np.pad(C, 1, mode='linear_ramp')
        intI += 1
        intJ += 1

    # local laplacian
    dC_ii = -(C[intI-1,intJ] + C[intI+1,intJ] - 2*C[intI,intJ])
    dC_jj = -(C[intI,intJ-1] + C[intI,intJ+1] - 2*C[intI,intJ])
    dC_ij = (C[intI+1,intJ+1] + C[intI-1,intJ-1]) -\
            (C[intI+1,intJ-1] -C[intI-1,intJ+1])

    dC_ii /= 4
    dC_jj /= 4
    dC_ij /= 8

    C_noise = np.maximum(1-C[intI,intJ], 0.)

    ψ = dC_ij**2 - dC_ii*dC_jj
    denom = ψ**2
    cov_ii = np.divide(-C_noise * ψ * dC_ii +
                       C_noise**2 * (dC_ii**2 + dC_ij**2),
                       denom, out=np.zeros_like(denom), where=denom!=0)
    cov_jj = np.divide(-C_noise * ψ * dC_jj +
                       C_noise**2 * (dC_jj**2 + dC_ij**2),
                       denom, out=np.zeros_like(denom), where=denom!=0)
    cov_ij = np.divide((C_noise * ψ -
                       C_noise**2 * (dC_ii + dC_jj)) * dC_ij,
                       denom, out=np.zeros_like(denom), where=denom!=0)
    return cov_ii, cov_jj, cov_ij

def gauss_spread(C, intI, intJ, dI, dJ, est='dist'):
    """ estimate an oriented gaussian function through the vicinity of the
    correlation function

    Parametes
    ---------
    C : numpy.ndarray, size=(m,n), dtype=float
        array with correlation values
    intI : integer, {x ∈ ℕ | x ≥ 0}
        location in rows of highest value in search space
    intJ : integer, {x ∈ ℕ | x ≥ 0}
        location in collumns of highest value in search space
    dI : float
        sub-pixel bias of top location along the row axis
    dJ : float
        sub-pixel bias of top location along the collumn axis
    est : string, default='dist'
        - 'dist' do weighted least sqaures dependent on the distance from the
          top location
        otherwise : ordinary least squares

    Returns
    -------
    cov_ii, cov_jj : float
        standard deviation of the vertical and horizontal axis
    ρ : float
        orientation of the Gaussian
    hess : numpy.ndarray, size=(4)
        estimate of the least squares computation
    frac :
        scaling of the correlation peak

    Notes
    -----
    The image frame is used in this function, hence the difference is:

        .. code-block:: text

          coordinate |              +--------> collumns
          system 'ij'|              |
                     |              |
                     |       j      |  image frame
             --------+-------->     |
                     |              |
                     |              v
                     | i            rows
                     v

    References
    ----------
    .. [1] Altena et al. "Correlation dispersion as a measure to better estimate
       uncertainty of remotely sensed glacier displacements" The cryosphere,
       vol.16(6) pp.2285-2300, 2021.
    """
    assert type(C) == np.ndarray, ('please provide an array')

    (m, n) = C.shape
    C -= np.mean(C)

    if np.min((intI, intJ)) <= 1 or (intI + 2) >= m or (intJ + 2) >= n:
        dub = 1.
    else:
        dub = 2.

    # is at the border
    if (intI == 0) or (intI + 1 == C.shape[0]) or \
            (intJ == 0) or (intJ + 1 == C.shape[1]):
        C = np.pad(C, 1, mode='linear_ramp')
        intI += 1
        intJ += 1

    I, J = np.mgrid[-dub:+dub + 1, -dub:+dub + 1]
    I -= dI
    J -= dJ

    dub = int(dub)
    P_sub = C[intI - dub:intI + dub + 1, intJ - dub:intJ + dub + 1]
    IN = P_sub > 0
    # normalize correlation score to probability function
    frac = np.sum(P_sub[IN])
    if frac != 0: P_sub /= frac

    if np.sum(IN) <=4:  # not enough data points
        return 0, 0, 0, np.zeros((4)), 0

    A = np.vstack((I[IN] ** 2,
                   2 * I[IN] * J[IN],
                   J[IN] ** 2,
                   np.ones((1, np.sum(IN)))
                   )).transpose()
    y = P_sub[IN]

    # least squares estimation
    if est == 'dist':
        dub = float(dub)
        W = np.abs(dub ** 2 - np.sqrt(A[:, 0] + A[:, 2])) / dub ** 2  # distance from top
        Aw = A * np.sqrt(W[:, np.newaxis])
        yw = y * np.sqrt(W)
        try:
            hess = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        except:
            hess = np.zeros(4)
    else:
        try:
            hess = np.linalg.lstsq(A, y, rcond=None)[0]
        except:
            hess = np.zeros(4)

    # convert to parameters
    denom = np.sqrt(np.abs(hess[0] * hess[2]))
    ρ = np.divide(0.5 * hess[1] , denom, where=denom!=0)
    cov_ii = -2 * (1 - ρ) * hess[0]
    if cov_ii!=0: cov_ii = np.divide(1, cov_ii)
    cov_jj = -2 * (1 - ρ) * hess[2]
    if cov_jj!=0: cov_jj = np.divide(1, cov_jj)

    # deviations can be negative
    if np.iscomplex(ρ) or np.iscomplex(cov_ii) or np.iscomplex(cov_jj):
        return 0, 0, 0
    return cov_ii, cov_jj, ρ, hess, frac

def intensity_disparity(I1,I2):
    """

    Parameters
    ----------
    I1 : numpy.ndarray, size=(m,n), dtype=float
        template with intensities
    I2 : numpy.ndarray, size=(m,n), dtype=float
        template with intensities

    Returns
    -------
    sigma_dI : float
        temporal intensity variation

    References
    ----------
    .. [1] Sciacchitano et al. "PIV uncertainty quantification by image
       matching" Measurement science and technology, vol.24 pp.045302, 2013.
    """
    assert type(I1) == np.ndarray, ('please provide an array')
    assert type(I2) == np.ndarray, ('please provide an array')
    are_two_arrays_equal(I1, I2)

    I1, I2 = I1.flatten(), I2.flatten()

    dI = I1 - I2
    # higher intensities have a larger contribution
    pI = np.sqrt(np.multiply( I1, I2))
    mu_dI = np.average(dI, weights=pI)
    sigma_dI = np.average((dI - mu_dI)**2, weights=pI)
    return sigma_dI
