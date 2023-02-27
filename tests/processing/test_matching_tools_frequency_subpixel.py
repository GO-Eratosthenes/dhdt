import numpy as np

from dhdt.processing.matching_tools_frequency_filters import \
    construct_phase_plane
from dhdt.processing.matching_tools_frequency_correlators import phase_corr
from dhdt.processing.matching_tools_frequency_subpixel import \
    phase_pca, phase_tpss, phase_difference, phase_ransac, \
    phase_lsq, phase_radon, phase_hough, phase_gradient_descend,\
    phase_weighted_pca, phase_secant
from dhdt.testing.matching_tools import test_subpixel_localization, \
    create_sample_image_pair

def test_phase_pca(d=2**5, max_range=1., tolerance=.1):
    im1,im2,di,dj,_ = create_sample_image_pair(d=d, max_range=max_range,
                                               integer=False)
    # perfect data
    Q = construct_phase_plane(im1,di,dj)
    di_pca,dj_pca,_,_ = phase_pca(Q)
    test_subpixel_localization(di_pca, dj_pca, di, dj, tolerance=tolerance)

    # real data
    Q = phase_corr(im1, im2)
    di_pca,dj_pca,_,_ = phase_pca(Q)
    test_subpixel_localization(di_pca, dj_pca, di, dj, tolerance=tolerance)
    return