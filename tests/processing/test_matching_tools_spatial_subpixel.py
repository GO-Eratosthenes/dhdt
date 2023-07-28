import numpy as np

from dhdt.testing.matching_tools import construct_correlation_peak, \
    _test_subpixel_localization

from dhdt.processing.matching_tools_spatial_subpixel import \
    get_top_centroid, get_top_moment, get_top_blais, get_top_esinc, \
    get_top_mass, get_top_ren, get_top_2d_gaussian, get_top_birchfield, \
    get_top_equiangular, get_top_gaussian, get_top_parabolic, \
    get_top_paraboloid, get_top_triangular


def test_get_top_centroid(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_centroid(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_moment(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    for d in np.arange(1, 3):
        ddi, ddj, i_int, j_int = get_top_moment(C, ds=d)
        di_hat, dj_hat = i_int + ddi, j_int + ddj
        _test_subpixel_localization(di_hat,
                                    dj_hat,
                                    di,
                                    dj,
                                    tolerance=tolerance)
    return


def test_get_top_esinc(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    for d in np.arange(1, 3):
        ddi, ddj, i_int, j_int = get_top_esinc(C, ds=d)
        di_hat, dj_hat = i_int + ddi, j_int + ddj
        _test_subpixel_localization(di_hat,
                                    dj_hat,
                                    di,
                                    dj,
                                    tolerance=tolerance)
    return


def test_get_top_blais(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_blais(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_mass(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_mass(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_ren(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_ren(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_2d_gaussian(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_2d_gaussian(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_birchfield(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_birchfield(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_equiangular(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_equiangular(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_gaussian(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_gaussian(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_parabolic(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_parabolic(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_paraboloid(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_paraboloid(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return


def test_get_top_triangular(ssize=(2 ** 4, 2 ** 4), tolerance=.4):
    di = np.random.random() * ssize[0] - ssize[0] // 2
    dj = np.random.random() * ssize[1] - ssize[1] // 2

    C = construct_correlation_peak(np.zeros(ssize), di, dj)
    ddi, ddj, i_int, j_int = get_top_triangular(C)
    di_hat, dj_hat = i_int + ddi, j_int + ddj
    _test_subpixel_localization(di_hat, dj_hat, di, dj, tolerance=tolerance)
    return
