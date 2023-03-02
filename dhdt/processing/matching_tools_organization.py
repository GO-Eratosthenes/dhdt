# organizational functions

import numpy as np

from .matching_tools_frequency_filters import \
    perdecomp, thresh_masking, coherence_masking
from .matching_tools_frequency_correlators import \
    cosi_corr, phase_only_corr, symmetric_phase_corr, amplitude_comp_corr, \
    orientation_corr, phase_corr, cross_corr, masked_cosine_corr, \
    binary_orientation_corr, masked_corr, robust_corr, windrose_corr, \
    gaussian_transformed_phase_corr, upsampled_cross_corr, \
    projected_phase_corr, gradient_corr, normalized_gradient_corr
from .matching_tools_frequency_subpixel import \
    phase_tpss, phase_svd, phase_radon, phase_hough, phase_ransac, \
    phase_weighted_pca, phase_pca, phase_lsq, phase_difference
from .matching_tools_frequency_metrics import \
    list_phase_metrics, get_phase_metric
from .matching_tools_spatial_correlators import \
    normalized_cross_corr, sum_sq_diff, sum_sad_diff, cumulative_cross_corr, \
    maximum_likelihood, weighted_normalized_cross_correlation
from .matching_tools_spatial_subpixel import \
    get_top_gaussian, get_top_parabolic, get_top_moment, \
    get_top_mass, get_top_centroid, get_top_blais, get_top_ren, \
    get_top_birchfield, get_top_equiangular, get_top_triangular, \
    get_top_esinc, get_top_paraboloid, get_top_2d_gaussian
from .matching_tools_spatial_metrics import \
    list_matching_metrics, get_correlation_metric, hessian_spread, gauss_spread
from .matching_tools_differential import \
    affine_optical_flow, hough_optical_flow

# admin
def list_frequency_correlators(unique=False):
    """ list the abbreviations of the different implemented correlators

    The following correlators are implemented:

        * cosi_corr : cosicorr
        * phas_only : phase only correlation
        * symm_phas : symmetric phase correlation
        * ampl_comp : amplitude compensation phase correlation
        * orie_corr : orientation correlation
        * grad_corr : gradient correlation
        * grad_norm : normalize gradient correlation
        * bina_phas : binary phase correlation
        * wind_corr : windrose correlation
        * gaus_phas : gaussian transformed phase correlation
        * cros_corr : cross correlation
        * robu_corr : robust phase correlation
        * proj_phas : projected phase correlation
        * phas_corr : phase correlation

    Returns
    -------
    correlator_list : list of strings
    """
    correlator_list = ['cosi_corr', 'phas_only', 'symm_phas', 'ampl_comp',
                       'orie_corr', 'grad_corr', 'grad_norm', 'bina_phas',
                       'wind_corr', 'gaus_phas', 'cros_corr', 'robu_corr',
                       'proj_phas', 'phas_corr']
    if unique:
        aka_list = []
    else:
        aka_list = ['CC',           # cross-correlation
                    'OC',           # orientation correlation
                    'PC',           # phase-only correlation
                    'GC',           # gradient correlation
                    'RC',           # robust correlation
                    'SPOF', 'SCOT', # symmetric phase correlation
                    'ACMF',         # amplitude compensation phase correlation
                    ]
        aka_list = [x.lower() for x in aka_list]
    return correlator_list + aka_list

def list_spatial_correlators(unique=False):
    """ list the abbreviations of the different implemented correlators.

    The following methods are implemented:

        * norm_corr : normalized cross correlation
        * sq_diff : sum of squared differences
        * sad_diff : sum of absolute differences
        * max_like : maximum likelihood
        * wght_corr : weighted normalized cross correlation

    Returns
    -------
    correlator_list : list of strings

    """
    correlator_list = ['norm_corr', 'sq_diff', 'sad_diff',
                       'max_like', 'wght_corr']
    if unique:
        aka_list = []
    else:
        aka_list = ['ncc',       # 'normalized cross correlation',
                    'wncc'       # weighted normalized cross correlation
                   ]
        aka_list = [x.lower() for x in aka_list]
    return correlator_list + aka_list

def list_differential_correlators():
    """" list the abbreviations of the different implemented differential
    correlators.

    The following are implemented:

        * lucas_kan
        * lucas_aff
        * hough_opt_flw

    Returns
    -------
    correlator_list : list of strings

    See Also
    --------
    list_phase_estimators, list_peak_estimators
    """
    correlator_list = ['lucas_kan', 'lucas_aff', 'hough_opt_flw']
    return correlator_list

def list_phase_estimators():
    """ list the abbreviations of the different implemented phase plane
    estimation procedures

    The following methods are implemented:

        * 'tpss' : two point step size
        * 'svd' : single value decomposition
        * 'radon' : radon transform
        * 'hough' : hough transform
        * 'ransac' : random sampling and consensus
        * 'wpca' : weighted principle component analysis
        * 'pca' : principle component analysis
        * 'lsq' : least squares estimation
        * 'diff' : phase difference

    Returns
    -------
    subpix_list : list of strings


    See Also
    --------
    list_peak_estimators, list_differential_correlators
    """
    subpix_list = ['tpss','svd','radon', 'hough', 'ransac', 'wpca',
                   'pca', 'lsq', 'diff']
    return subpix_list

def list_peak_estimators():
    """ list the abbreviations of the different implemented for the peak fitting
    of the similarity function.

    The following methods are implemented:

        * 'gauss_1' : 1D Gaussian fitting
        * 'parab_1' : 1D parabolic fitting
        * 'moment' : 2D moment of the peak
        * 'mass' : 2D center of mass fitting
        * 'centroid' : 2D estimate the centroid
        * 'blais' : 1D estimate through forth order filter
        * 'ren' : 1D parabolic fitting
        * 'birch' : 1D weighted sum
        * 'eqang' : 1D equiangular line fitting
        * 'trian' : 1D triangular fitting
        * 'esinc' : 1D exponential esinc function
        * 'gauss_2' : 2D Gaussian fitting
        * 'parab_2' : 2D parabolic fitting
        * 'optical_flow' : optical flow refinement

    Returns
    -------
    subpix_list : list of strings

    See Also
    --------
    list_phase_estimators, list_differential_correlators
    """
    subpix_list = ['gauss_1', 'parab_1', 'moment', 'mass', 'centroid',
                  'blais', 'ren', 'birch', 'eqang', 'trian', 'esinc',
                  'gauss_2', 'parab_2', 'optical_flow']
    return subpix_list

# todo: include masks
def estimate_translation_of_two_subsets(I1, I2, M1, M2, correlator='lucas_kan',
                                        **kwargs):
    """

    Parameters
    ----------
    I1,I2 : numpy.ndarray, size=(m,n), dtype={float,integer}
        grids with intensities
    M1,M2 : numpy.ndarray, size=(m,n), dtype=bool
        labelled array corresponding to I1,I2. Hence True denotes exclusion.
    correlator : string, default='lucas_kanade'
        methodology to use to correlate I1 and I2

    Returns
    -------
    di, dj : float, unit=pixels
        relative displacement (translation) between I1 and I2
    score : float
        (dis)-similarity score
    """
    assert type(I1) == np.ndarray, ('please provide an array')
    assert type(I2) == np.ndarray, ('please provide an array')
    assert type(M1) == np.ndarray, ('please provide an array')
    assert type(M2) == np.ndarray, ('please provide an array')

    optical_flow_approaches = list_differential_correlators()
    assert (correlator in optical_flow_approaches), \
            ('please provide a valid optical flow approach. ' +
             'this can be one of the following:'+
             f' { {*optical_flow_approaches} }')

    if correlator in ['lucas_aff']:
        di,dj,_,score = affine_optical_flow(I1, I2, model='affine',
                                            preprocessing=
                                            kwargs.get('preprocessing'))
    if correlator in ['hough_opt_flw']:
        num_est, max_amp = 1, 1
        if kwargs.get('num_estimates') != None:
            num_est = kwargs.get('num_estimates')
        if kwargs.get('max_amp') != None:
            max_amp = kwargs.get('max_amp')

        if M1.size!=0: I1[M1] = np.nan # set data outside mask to NaN
        if M2.size!=0: I2[M2] = np.nan  # set data outside mask to NaN

        di,dj,score = hough_optical_flow(I1, I2,
                                         num_estimates=num_est,
                                         preprocessing=kwargs.get('preprocessing'),
                                         max_amp=max_amp)
    else: #'lucas_kan'
        di,dj,_,score = affine_optical_flow(I1, I2, model='simple',
                                            preprocessing=
                                            kwargs.get('preprocessing')
                                            )

    return di, dj, score

def match_translation_of_two_subsets(I1_sub,I2_sub,correlator,subpix,
                                     M1_sub=np.array([]), M2_sub=np.array([]) ):
    """

    Parameters
    ----------
    I1_sub : numpy.ndarray, size={(m,n),(k,l)}, dtype={float,integer}, ndim={2,3}
        grid with intensities, a.k.a. template to locate
    I2_sub : numpy.ndarray, size=(m,n), dtype={float,integer}, ndim={2,3}
        grid with intensities, a.k.a. search space
    correlator : string
        methodology to use to correlate I1 and I2
    subpix : string
        methodology to refine the to sub-pixel localization
    M1_sub,M2_sub :  numpy.ndarray, size=(m,n), dtype=bool
        labels corresponding to I1_sub,I2_sub. Hence True denotes exclusion.

    Returns
    -------
    {Q,C} : numpy.ndarray, size={(m,n),(k,l)}, dtype={complex,float}, ndim=2
        cross-correlation spectra or function

    See Also
    --------
    estimate_subpixel, estimate_precision
    list_frequency_correlators, list_spatial_correlators,
    list_phase_estimators, list_peak_estimators

    """
    assert type(I1_sub)==np.ndarray, ('please provide an array')
    assert type(I2_sub)==np.ndarray, ('please provide an array')
    assert type(M1_sub)==np.ndarray, ('please provide an array')
    assert type(M2_sub)==np.ndarray, ('please provide an array')
    frequency_based = list_frequency_correlators()
    spatial_based = list_spatial_correlators()

    # some sub-pixel methods use the peak of the correlation surface,
    # while others need the phase of the cross-spectrum
    phase_based = list_phase_estimators()
    peak_based = list_peak_estimators()

    assert ((correlator in frequency_based) or
        (correlator in spatial_based)), ('please provide a valid correlation '+
                                     'method. it can be one of the following:'+
                                     f' { {*frequency_based,*spatial_based} }')

    # reduce edge effects in frequency space
    if correlator in frequency_based:
        I1_sub,I2_sub = perdecomp(I1_sub)[0], perdecomp(I2_sub)[0]

    # translational matching/estimation
    if correlator in frequency_based:
        # frequency correlator
        if correlator in ['cosi_corr']:
            Q = cosi_corr(I1_sub, I2_sub)[0]
        elif correlator in ['phas_corr', 'pc']:
            Q = phase_corr(I1_sub, I2_sub)
        elif correlator in ['phas_only']:
            Q = phase_only_corr(I1_sub, I2_sub)
        elif correlator in ['symm_phas', 'spof', 'scot']:
            Q = symmetric_phase_corr(I1_sub, I2_sub)
        elif correlator in ['ampl_comp', 'acmf']:
            Q = amplitude_comp_corr(I1_sub, I2_sub)
        elif correlator in ['orie_corr', 'oc']:
            Q = orientation_corr(I1_sub, I2_sub)
        elif correlator in ['grad_corr', 'gc']:
            Q = gradient_corr(I1_sub, I2_sub)
        elif correlator in ['grad_norm']:
            Q = normalized_gradient_corr(I1_sub, I2_sub)
        elif correlator in ['mask_corr']:
            C = masked_corr(I1_sub, I2_sub, M1_sub, M2_sub)
            if subpix in phase_based: Q = np.fft.fft2(C)
        elif correlator in ['bina_phas']:
            Q = binary_orientation_corr(I1_sub, I2_sub)
        elif correlator in ['wind_corr']:
            Q = windrose_corr(I1_sub, I2_sub)
        elif correlator in ['gaus_phas']:
            Q = gaussian_transformed_phase_corr(I1_sub, I2_sub)
        elif correlator in ['upsp_corr']:
            Q = upsampled_cross_corr(I1_sub, I2_sub)
        elif correlator in ['cros_corr', 'cc']:
            Q = cross_corr(I1_sub, I2_sub)
        elif correlator in ['robu_corr', 'rc']:
            Q = robust_corr(I1_sub, I2_sub)
        elif correlator in ['proj_phas']:
            C = projected_phase_corr(I1_sub, I2_sub, M1_sub, M2_sub)
            if subpix in phase_based: Q = np.fft.fft2(C)
        if (subpix in peak_based) and ('Q' in locals()):
            C = np.fft.fftshift(np.real(np.fft.ifft2(Q)))
    else:
        # spatial correlator
        if correlator in ['norm_corr', 'ncc']:
            C = normalized_cross_corr(I1_sub, I2_sub)
        elif correlator in ['sq_diff']:
            C = 1 - sum_sq_diff(I1_sub, I2_sub)
        elif correlator in ['sad_diff']:
            C = 1 - sum_sad_diff(I1_sub, I2_sub)
        elif correlator in ['max_like']:
            C = maximum_likelihood(I1_sub, I2_sub)
        elif correlator in ['wght_corr', 'wncc']:
            C = weighted_normalized_cross_correlation(I1_sub, I2_sub)

        if subpix in phase_based:
            Q = np.fft.fft2(C)

    if (subpix in peak_based) or (subpix is None):
        return C
    else:
        return Q

def estimate_subpixel(QC, subpix, m0=np.zeros((1, 2)), **kwargs):
    """ estimate the sub-pixel translational displacement, present in a
    cross-correlation function or spectra.

    Parameters
    ----------
    QC : numpy.ndarray, size=(m,n), dtype={complex,float}, ndim=2
        cross-correlation spectra or function
    subpix : string
        methodology to refine the to sub-pixel localization
    m0 : numpy.ndarray, size=(1,2), dtype=integer
        row, collumn location of the peak of the cross-correlation function

    Returns
    -------
    ddi,ddj : float
        sub-pixel location of the peak/phase-plane

    See Also
    --------
    match_translation_of_two_subsets,
    list_phase_estimators, list_peak_estimators
    """
    assert type(QC)==np.ndarray, ('please provide an array')
    assert type(m0)==np.ndarray, ('please provide an array')
    phase_based = list_phase_estimators()
    peak_based = list_peak_estimators()
    assert ((subpix in phase_based) or (subpix in peak_based)), \
        ('please provide a valid subpixel method.' +
         'it can be one of the following:' +
         f' { {*peak_based,*phase_based} }')

    ds = 1
    if kwargs.get('ds') != None: ds = kwargs.get('num_estimates')

    if subpix in peak_based: # correlation surface
        if subpix in ['gauss_1']:
            ddi,ddj,_,_ = get_top_gaussian(QC, top=m0)
        elif subpix in ['parab_1']:
            ddi,ddj,_,_= get_top_parabolic(QC, top=m0)
        elif subpix in ['moment']:
            ddi,ddj,_,_= get_top_moment(QC, ds=ds, top=m0)
        elif subpix in ['mass']:
            ddi,ddj,_,_= get_top_mass(QC, top=m0)
        elif subpix in ['centroid']:
            ddi,ddj,_,_= get_top_centroid(QC, top=m0)
        elif subpix in ['blais']:
            ddi,ddj,_,_= get_top_blais(QC, top=m0)
        elif subpix in ['ren']:
            ddi,ddj,_,_= get_top_ren(QC, top=m0)
        elif subpix in ['birch']:
            ddi,ddj,_,_= get_top_birchfield(QC, top=m0)
        elif subpix in ['eqang']:
            ddi,ddj,_,_= get_top_equiangular(QC, top=m0)
        elif subpix in ['trian']:
            ddi,ddj,_,_= get_top_triangular(QC, top=m0)
        elif subpix in ['esinc']:
            ddi,ddj,_,_= get_top_esinc(QC, ds=ds, top=m0)
        elif subpix in ['gauss_2']:
            ddi,ddj,_,_= get_top_2d_gaussian(QC, ds=ds, top=m0)
        elif subpix in ['parab_2t']:
            ddi,ddj,_,_= get_top_paraboloid(QC, ds=ds, top=m0)

    elif subpix in phase_based: #cross-spectrum
        if subpix in ['tpss']:
            W = thresh_masking(QC)
            ddi,ddj,snr = phase_tpss(QC, W, m0)
        elif subpix in ['svd']:
            W = coherence_masking(QC)
            ddi,ddj = phase_svd(QC, W, rad=0.4)
        elif subpix in ['radon']:
            ddi,ddj = phase_radon(QC)
        elif subpix in ['hough']:
            ddi,ddj = phase_hough(QC)
        elif subpix in ['ransac']:
            ddi,ddj = phase_ransac(QC)
        elif subpix in ['wpca']:
            W = thresh_masking(QC)
            ddi,ddj = phase_weighted_pca(QC, W)
        elif subpix in ['pca']:
            ddi,ddj = phase_pca(QC)
        elif subpix in ['lsq']:
            ddi,ddj = phase_lsq(QC)
        elif subpix in ['diff']:
            ddi,ddj = phase_difference(QC)

    return ddi, ddj

def estimate_match_metric(QC, di=None, dj=None, metric='snr'):
    peak_metric = list_matching_metrics()
    phase_metric = list_phase_metrics()

    if metric in peak_metric:
        score = get_correlation_metric(QC, metric=metric)
    elif metric in phase_metric:
        score = get_phase_metric(QC, di, dj, metric=metric)
    return score

def estimate_precision(C, di, dj, method='gaussian'):
    """ given a similarity surface, estimate its matching dispersion.
    See for a full describtion [Al21]_

    Parameters
    ----------
    C : numpy.ndarray, size=(m,n)
        cross correlation function
    di,dj : float
        locational estimate of the cross-correlation peak, this can also be
        negative, see Notes below
    method : {'gaussian', 'hessian'}

    Returns
    -------
    cov_ii, cov_jj : float
        co-variance estimate
    cov_ij : float
        off-diagonal co-variance estimate

    Notes
    -----
    It is important to know what type of coordinate systems exist, hence:

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

    See Also
    --------
    estimate_subpixel, match_translation_of_two_subsets
    hessian_spread, gauss_spread

    References
    ----------
    .. [Al21] Altena et al. "Correlation dispersion as a measure to better
              estimate uncertainty of remotely sensed glacier displacements"
              The cryosphere, vol.16(6) pp.2285-2300, 2021.
    """
    if method in ['hessian']:
        cov_ii,cov_jj,cov_ij = hessian_spread(C,
           C.shape[0] // 2 + np.round(di).astype(int),
           C.shape[1] // 2 + np.round(dj).astype(int))
    else:
        cov_ii,cov_jj,cov_ij,_,_ = gauss_spread(C,
            C.shape[0]//2 + np.round(di).astype(int),
            C.shape[1]//2 + np.round(dj).astype(int),
            di-np.round(di), dj-np.round(dj), est='dist')
    return cov_ii, cov_jj, cov_ij