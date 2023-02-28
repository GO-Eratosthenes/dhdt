import numpy as np

from dhdt.processing.matching_tools_frequency_correlators import \
    phase_corr, phase_only_corr, robust_corr, orientation_corr, \
    cross_corr, symmetric_phase_corr, amplitude_comp_corr, \
    upsampled_cross_corr, projected_phase_corr

from dhdt.testing.matching_tools import \
    create_sample_image_pair, test_phase_plane_localization

def test_phase_corr(d=2**5, tolerance=.4):
    for bands in [1, 3]:  # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=np.inf,
                                                   integer=True, ndim=bands)
        Q = phase_corr(im1, im2)
        test_phase_plane_localization(Q, di, dj, tolerance=tolerance)
    return

def test_phase_only_corr(d=2**5, tolerance=.4):
    for bands in [1, 3]:  # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=np.inf,
                                                   integer=True, ndim=bands)
        Q = phase_only_corr(im1, im2)
        test_phase_plane_localization(Q, di, dj, tolerance=tolerance)
    return

def test_robust_corr(d=2**5, tolerance=.4):
    for bands in [1,3]: # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=np.inf,
                                                   integer=True, ndim=bands)
        Q = robust_corr(im1, im2)
        test_phase_plane_localization(Q, di, dj, tolerance=tolerance)
    return

def test_orientation_corr(d=2**5, tolerance=.4):
    for bands in [1, 3]:  # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=np.inf,
                                                   integer=True, ndim=bands)
        Q = orientation_corr(im1, im2)
        test_phase_plane_localization(Q, di, dj, tolerance=tolerance)
    return

def test_cross_corr(d=2**5, tolerance=.4):
    for bands in [1, 3]:  # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=np.inf,
                                                   integer=True, ndim=bands)
        Q = cross_corr(im1, im2)
        test_phase_plane_localization(Q, di, dj, tolerance=tolerance)
    return

def test_symmetric_phase_corr(d=2**5, tolerance=.4):
    for bands in [1, 3]:  # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=np.inf,
                                                   integer=True, ndim=bands)
        Q = symmetric_phase_corr(im1, im2)
        test_phase_plane_localization(Q, di, dj, tolerance=tolerance)
    return

def test_amplitude_comp_corr(d=2**5, tolerance=.4):
    for bands in [1, 3]:  # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=np.inf,
                                                   integer=True, ndim=bands)
        Q = amplitude_comp_corr(im1, im2)
        test_phase_plane_localization(Q, di, dj, tolerance=tolerance)
    return

def test_upsampled_cross_corr(d=2**5, tolerance=.1):
    for bands in [1, 3]:  # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=np.inf,
                                                   integer=True, ndim=bands)
        Q = upsampled_cross_corr(im1, im2)
        test_phase_plane_localization(Q, di, dj, tolerance=tolerance)
    return

def test_projected_phase_corr(d=2**5, tolerance=.1):
    for bands in [1, 3]:  # testing both grayscale and multi-spectral data
        im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=np.inf,
                                                   integer=True, ndim=bands)
        Q = projected_phase_corr(im1, im2)
        test_phase_plane_localization(Q, di, dj, tolerance=tolerance)
    return
