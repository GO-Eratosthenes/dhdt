# organizational functions

import numpy as np

from .matching_tools_frequency_filters import \
    perdecomp, thresh_masking
from .matching_tools_frequency_correlators import \
    cosi_corr, phase_only_corr, symmetric_phase_corr, amplitude_comp_corr, \
    orientation_corr, phase_corr, cross_corr, masked_cosine_corr, \
    binary_orientation_corr, masked_corr, robust_corr, windrose_corr, \
    gaussian_transformed_phase_corr, upsampled_cross_corr, \
    projected_phase_corr
from .matching_tools_frequency_subpixel import \
    phase_tpss, phase_svd, phase_radon, phase_hough, phase_ransac, \
    phase_weighted_pca, phase_pca, phase_lsq, phase_difference
from .matching_tools_spatial_correlators import \
    normalized_cross_corr, sum_sq_diff, sum_sad_diff, cumulative_cross_corr, \
    maximum_likelihood, weighted_normalized_cross_correlation
from .matching_tools_spatial_subpixel import \
    get_top_gaussian, get_top_parabolic, get_top_moment, \
    get_top_mass, get_top_centroid, get_top_blais, get_top_ren, \
    get_top_birchfield, get_top_equiangular, get_top_triangular, \
    get_top_esinc, get_top_paraboloid, get_top_2d_gaussian
from .matching_tools_correlation_metrics import \
    hessian_spread, gauss_spread
from .matching_tools_differential import \
    affine_optical_flow, hough_optical_flow

# admin
def list_frequency_correlators():
    """ list the abbreviations of the different implemented correlators, being:
        cosi_corr - cosicorr
        phas_only - phase only correlation
        symm_phas - symmetric phase correlation
        ampl_comp - amplitude compensation phase correlation
        orie_corr - orientation correlation
        mask_corr - masked normalized cross correlation
        bina_phas - binary phase correlation
        wind_corr - windrose correlation
        gaus_phas - gaussian transformed phase correlation
        upsp_corr - upsampled cross correlation
        cros_corr - cross correlation
        robu_corr - robust phase correlation
        proj_phas - projected phase correlation
        phas_corr - phase correlation
    """
    correlator_list = ['cosi_corr', 'phas_only', 'symm_phas', 'ampl_comp',
                       'orie_corr', 'mask_corr', 'bina_phas', 'wind_corr',
                       'gaus_phas', 'upsp_corr', 'cros_corr', 'robu_corr',
                       'proj_phas', 'phas_corr']
    return correlator_list

def list_spatial_correlators():
    """ list the abbreviations of the different implemented correlators, being:
        norm_corr - normalized cross correlation
        cumu_corr - cumulative cross correlation
        sq_diff - sum of squared differences
        sad_diff - sum of absolute differences
        max_like - phase correlation
        wght_corr - weighted normalized cross correlation
    """
    correlator_list = ['norm_corr', 'cumu_corr', 'sq_diff', 'sad_diff',
                       'max_like', 'wght_corr']
    return correlator_list

def list_differential_correlators():
    correlator_list = ['lucas_kan', 'lucas_aff', 'hough_opt_flw']
    return correlator_list

def list_phase_estimators():
    """ list the abbreviations of the different implemented phase plane
    estimation procedures, being:
        * 'tpss' : two point step size
        * 'svd' : single value decomposition
        * 'radon' : radon transform
        * 'hough' : hough transform
        * 'ransac' : random sampling and consensus
        * 'wpca' : weighted principle component analysis
        * 'pca' : principle component analysis
        * 'lsq' : least squares estimation
        * 'diff' : phase difference
    """
    subpix_list = ['tpss','svd','radon', 'hough', 'ransac', 'wpca',\
                   'pca', 'lsq', 'diff']
    return subpix_list

def list_peak_estimators():
    """ list the abbreviations of the different implemented for the peak fitting
    of the similarity function, being:
        'gauss_1' : 1D Gaussian fitting
        'parab_1' : 1D parabolic fitting
        'moment' : 2D moment of the peak
        'mass' : 2D center of mass fitting
        'centroid' : 2D estimate the centroid
        'blais' : 1D estimate through forth order filter
        'ren' : 1D parabolic fitting
        'birch' : 1D weighted sum
        'eqang' : 1D equiangular line fitting
        'trian' : 1D triangular fitting
        'esinc' : 1D exponential esinc function
        'gauss_2' : 2D Gaussian fitting
        'parab_2' : 2D parabolic fitting
        'optical_flow' : optical flow refinement
    """
    subpix_list = ['gauss_1', 'parab_1', 'moment', 'mass', 'centroid',
                  'blais', 'ren', 'birch', 'eqang', 'trian', 'esinc',
                  'gauss_2', 'parab_2', 'optical_flow']
    return subpix_list

# todo: include masks
def estimate_translation_of_two_subsets(I1, I2, M1, M2, correlator='lucas_kan',
                                        **kwargs):
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

        if M1.size!=0: I1[~M1] = np.nan # set data outside mask to NaN
        if M2.size!=0: I2[~M2] = np.nan  # set data outside mask to NaN

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
                                     M1_sub=np.array([]),
                                     M2_sub=np.array([]) ):
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
        elif correlator in ['phas_corr']:
            Q = phase_corr(I1_sub, I2_sub)
        elif correlator in ['phas_only']:
            Q = phase_only_corr(I1_sub, I2_sub)
        elif correlator in ['symm_phas']:
            Q = symmetric_phase_corr(I1_sub, I2_sub)
        elif correlator in ['ampl_comp']:
            Q = amplitude_comp_corr(I1_sub, I2_sub)
        elif correlator in ['orie_corr']:
            Q = orientation_corr(I1_sub, I2_sub)
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
        elif correlator in ['cros_corr']:
            Q = cross_corr(I1_sub, I2_sub)
        elif correlator in ['robu_corr']:
            Q = robust_corr(I1_sub, I2_sub)
        elif correlator in ['proj_phas']:
            C = projected_phase_corr(I1_sub, I2_sub, M1_sub, M2_sub)
            if subpix in phase_based: Q = np.fft.fft2(C)
        if (subpix in peak_based) and ('Q' in locals()):
            C = np.fft.fftshift(np.real(np.fft.ifft2(Q)))
    else:
        # spatial correlator
        if correlator in ['norm_corr']:
            C = normalized_cross_corr(I1_sub, I2_sub)
        elif correlator in ['cumu_corr']:
            C = cumulative_cross_corr(I1_sub, I2_sub)
        elif correlator in ['sq_diff']:
            C = -1 * sum_sq_diff(I1_sub, I2_sub)
        elif correlator in ['sad_diff']:
            C = sum_sad_diff(I1_sub, I2_sub)
        elif correlator in ['max_like']:
            C = maximum_likelihood(I1_sub, I2_sub)
        elif correlator in ['wght_corr']:
            C = weighted_normalized_cross_correlation(I1_sub, I2_sub)
        if subpix in phase_based:
            Q = np.fft.fft2(C)

    if (subpix in peak_based) or (subpix is None):
        return C
    else:
        return Q

def estimate_subpixel(QC, subpix, m0=np.zeros((1,2))):
    assert type(QC)==np.ndarray, ('please provide an array')
    assert type(m0)==np.ndarray, ('please provide an array')
    phase_based = list_phase_estimators()
    peak_based = list_peak_estimators()
    assert ((subpix in phase_based) or (subpix in peak_based)), \
        ('please provide a valid subpixel method.' +
         'it can be one of the following:' +
         f' { {*peak_based,*phase_based} }')

    if subpix in peak_based: # correlation surface
        if subpix in ['gauss_1']:
            ddi,ddj,_,_ = get_top_gaussian(QC, top=m0)
        elif subpix in ['parab_1']:
            ddi,ddj,_,_= get_top_parabolic(QC, top=m0)
        elif subpix in ['moment']:
            ddi,ddj,_,_= get_top_moment(QC, ds=1, top=m0)
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
            ddi,ddj,_,_= get_top_esinc(QC, ds=1, top=m0)
        elif subpix in ['gauss_2']:
            ddi,ddj,_,_= get_top_2d_gaussian(QC, ds=1, top=m0)
        elif subpix in ['parab_2t']:
            ddi,ddj,_,_= get_top_paraboloid(QC, ds=1, top=m0)

    elif subpix in phase_based: #cross-spectrum
        if subpix in ['tpss']:
            W = thresh_masking(QC)
            (m,snr) = phase_tpss(QC, W, m0)
            ddi, ddj = m[0], m[1]
        elif subpix in ['svd']:
            W = thresh_masking(QC)
            ddi,ddj = phase_svd(QC, W)
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

def estimate_precision(C, di, dj, method='gaussian'):

    if method in ['hessian']:
        cov_ii,cov_jj,cov_ij = hessian_spread(C, np.round(di), np.round(dj))
    else:
        cov_ii,cov_jj,cov_ij,_,_ = gauss_spread(C,
                                                np.round(di).astype(int),
                                                np.round(dj).astype(int),
                                                di%1, dj%1, est='dist')
    return cov_ii, cov_jj, cov_ij