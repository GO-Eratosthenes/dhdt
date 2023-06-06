import numpy as np

from dhdt.processing.matching_tools_frequency_filters import \
    construct_phase_plane
from dhdt.processing.matching_tools_frequency_correlators import phase_corr
from dhdt.processing.matching_tools_frequency_subpixel import \
    phase_pca, phase_tpss, phase_difference, phase_ransac, \
    phase_lsq, phase_radon, phase_hough, phase_gradient_descend,\
    phase_weighted_pca, phase_secant
from dhdt.testing.matching_tools import _test_subpixel_localization, \
    create_sample_image_pair

# limited domain
def test_phase_pca(d=2**5, max_range=1., tolerance=.1):
    im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=max_range,
                                               integer=False)
    # perfect data
    Q, W = construct_phase_plane(im1,di,dj), np.ones_like(im1, dtype=bool)
    di_hat,dj_hat = phase_pca(Q, W)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)

    # real data
    Q = phase_corr(im1, im2)
    di_hat,dj_hat = phase_pca(Q)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return

def test_phase_tpss(d=2**5, max_range=1., tolerance=.1):
    im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=max_range,
                                               integer=False)
    # perfect data
    Q, W = construct_phase_plane(im1,di,dj), np.ones_like(im1, dtype=bool)
    m0 = np.random.rand(2)-.5
    di_hat,dj_hat,_ = phase_tpss(Q, W, m0)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)

    # real data
    Q = phase_corr(im1, im2)
    di_hat,dj_hat,_ = phase_tpss(Q)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return

def test_phase_lsq(d=2**5, max_range=1., tolerance=.1):
    im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=max_range,
                                               integer=False)
    # perfect data
    Q, W = construct_phase_plane(im1,di,dj), np.ones_like(im1, dtype=bool)
    di_hat,dj_hat = phase_lsq(Q, W)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)

    # real data
    Q = phase_corr(im1, im2)
    di_hat,dj_hat = phase_lsq(Q)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return

def test_gradient_descend(d=2**5, max_range=1., tolerance=.1):
    im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=max_range,
                                               integer=False)
    # perfect data
    Q, W = construct_phase_plane(im1,di,dj), np.ones_like(im1, dtype=bool)
    di_hat,dj_hat = phase_gradient_descend(Q, W)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)

    # real data
    Q = phase_corr(im1, im2)
    di_hat,dj_hat = phase_gradient_descend(Q)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


# extended domain, i.e.: more than one pixel displacement
def test_phase_difference(d=2**5, max_range=10., tolerance=.1):
    im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=max_range,
                                               integer=False)
    # perfect data
    Q, W = construct_phase_plane(im1,di,dj), np.ones_like(im1, dtype=bool)
    di_hat,dj_hat,_,_ = phase_difference(Q, W)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)

    # real data
    Q = phase_corr(im1, im2)
    di_hat,dj_hat,_,_ = phase_difference(Q)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return

def test_phase_ransac(d=2**5, max_range=10., tolerance=.1):
    im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=max_range,
                                               integer=False)
    # perfect data
    Q, W = construct_phase_plane(im1,di,dj), np.ones_like(im1, dtype=bool)
    di_hat,dj_hat,_,_ = phase_ransac(Q, W)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)

    # real data
    Q = phase_corr(im1, im2)
    di_hat,dj_hat,_,_ = phase_ransac(Q)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return

def test_phase_randon(d=2**5, max_range=10., tolerance=.1):
    im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=max_range,
                                               integer=False)
    # perfect data
    Q = construct_phase_plane(im1,di,dj)
    di_hat,dj_hat = phase_radon(Q)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)

    # real data
    Q = phase_corr(im1, im2)
    di_hat,dj_hat = phase_radon(Q)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return

def test_phase_hough(d=2**5, max_range=10., tolerance=.1):
    im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=max_range,
                                               integer=False)
    # perfect data
    Q, W = construct_phase_plane(im1,di,dj), np.ones_like(im1, dtype=bool)
    di_hat,dj_hat,_,_ = phase_hough(Q, W)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)

    # real data
    Q = phase_corr(im1, im2)
    di_hat,dj_hat,_,_ = phase_hough(Q)
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return
