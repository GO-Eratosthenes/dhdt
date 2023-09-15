import numpy as np

from dhdt.testing.matching_tools import create_sample_image_pair, \
    _test_subpixel_localization
from dhdt.processing.matching_tools_spatial_subpixel import \
    get_integer_peak_location

from dhdt.processing.matching_tools_spatial_correlators import \
    normalized_cross_corr, sum_sad_diff, sum_sq_diff, maximum_likelihood, \
    weighted_normalized_cross_correlation, cosine_similarity


def test_normalized_cross_corr(d=2**5, tolerance=.4):
    im1, im2, di, dj, im1_big = create_sample_image_pair(d=d,
                                                         max_range=np.inf,
                                                         integer=True,
                                                         ndim=1)
    C = normalized_cross_corr(im2, im1_big)
    di_hat, dj_hat = get_integer_peak_location(C)[:2]
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_sum_sad_diff(d=2**4, tolerance=.4):
    im1, im2, di, dj, im1_big = create_sample_image_pair(d=d,
                                                         max_range=np.inf,
                                                         integer=True,
                                                         ndim=1)
    C = 1 - sum_sad_diff(im2, im1_big)
    di_hat, dj_hat = get_integer_peak_location(C)[:2]
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_sum_sq_diff(d=2**5, tolerance=.4):
    im1, im2, di, dj, im1_big = create_sample_image_pair(d=d,
                                                         max_range=np.inf,
                                                         integer=True,
                                                         ndim=1)
    C = sum_sq_diff(im2, im1_big)
    di_hat, dj_hat = get_integer_peak_location(C)[:2]
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_maximum_likelihood(d=2**5, tolerance=.4):
    im1, im2, di, dj, im1_big = create_sample_image_pair(d=d,
                                                         max_range=np.inf,
                                                         integer=True,
                                                         ndim=1)
    C = maximum_likelihood(im2, im1_big)
    di_hat, dj_hat = get_integer_peak_location(C)[:2]
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_weighted_normalized_cross_corr(d=2**5, tolerance=.4):
    im1, im2, di, dj, im1_big = create_sample_image_pair(d=d,
                                                         max_range=np.inf,
                                                         integer=True,
                                                         ndim=1)
    C = weighted_normalized_cross_correlation(im2, im1_big)
    di_hat, dj_hat = get_integer_peak_location(C)[:2]
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_cosine_similarity(d=2**5, tolerance=.4):
    im1, im2, di, dj, im1_big = create_sample_image_pair(d=d,
                                                         max_range=np.inf,
                                                         integer=True,
                                                         ndim=1)

    H_x = get_grad_filters(ftype='kroon', tsize=3, order=1)[0]
    H_y = np.transpose(H_x)
    G1 = ndimage.convolve(im2, H_x) + 1j * ndimage.convolve(im2, H_y)
    G2 = ndimage.convolve(im1_big, H_x) + 1j * ndimage.convolve(im1_big, H_y)
    G1, G2 = normalize_power_spectrum(G1), normalize_power_spectrum(G2)
    C = cosine_similarity(im2, im1_big)
    di_hat, dj_hat = get_integer_peak_location(C)[:2]
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return
