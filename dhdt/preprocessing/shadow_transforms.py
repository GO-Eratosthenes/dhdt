import numpy as np

from dhdt.generic.data_tools import s_curve

from ..generic.unit_check import are_three_arrays_equal, are_two_arrays_equal
from .color_transforms import (erdas2hsi, lab2lch, lms2lab, rgb2hcv, rgb2hsi,
                               rgb2lms, rgb2xyz, rgb2ycbcr, rgb2yiq, xyz2lab)
from .multispec_transforms import (pca_rgb_preparation,
                                   principle_component_analysis)

def _get_shadow_transforms():
    methods_dict = {
        'siz' : 'shadow index first version',
        'isi' : 'improved shadow index',
        'sei' : 'shadow enhancement index',
        'fcsdi' : 'false color shadow difference index',
        'csi' : 'combinational shadow index',
        'nsvdi' : 'normalized saturation value difference index',
        'sil' : 'shadow index second',
        'sr' : 'specthem ratio',
        'sdi' : 'shadow detector index',
        'sisdi' : 'saturation intensity shadow detector index',
        'mixed' : 'mixed property based shadow index',
        'msf' : 'modified shadow fraction',
        'c3' : 'color invariant',
        'entropy' : 'entropy shade removal',
        'shi' : 'shade index',
        'sp' : 'shadow probabilities',
        'nri' : 'normalized range shadow index'
    }
    return methods_dict

def apply_shadow_transform(method, Blue, Green, Red, RedEdge, Near, Shw,
                           **kwargs):
    """ Given a specific method, employ shadow transform

    Parameters
    ----------
    method : {‘siz’,’ isi’,’ sei’,’ fcsdi’, ‘nsvi’, ‘nsvdi’, ‘sil’, ‘sr’,\
              ‘sdi’, ‘sisdi’, ‘mixed’, ‘c3’, ‘entropy’, ‘shi’,’ sp’, 'nri'}
        method name to be implemented,can be one of the following:

        * 'siz' : shadow index first version
        * 'isi' : improved shadow index
        * 'sei' : shadow enhancement index
        * 'fcsdi' : false color shadow difference index
        * 'csi' : combinational shadow index
        * 'nsvdi' : normalized saturation value difference index
        * 'sil' : shadow index second
        * 'sr' : specthem ratio
        * 'sdi' : shadow detector index
        * 'sisdi' : saturation intensity shadow detector index
        * 'mixed' : mixed property based shadow index
        * 'msf' : modified shadow fraction
        * 'c3' : color invariant
        * 'entropy' : entropy shade removal
        * 'shi' : shade index
        * 'sp' : shadow probabilities
        * 'nri' : normalized range shadow index
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    RedEdge : numpy.ndarray, size=(m,n)
        red edge band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near-infrared band of satellite image
    Shw : numpy.ndarray, size=(m,n)
        synthetic shading from for example a digital elevation model

    Returns
    -------
    M : numpy.ndarray, size=(m,n)
        shadow enhanced satellite image

    See Also
    --------
    .handler_multispec.get_shadow_bands :
        provides Blue, Green, Red bandnumbers for a specific satellite
    .handler_multispec.read_shadow_bands :
        reads the specific bands given
    """
    method = method.lower()
    methods = _get_shadow_transforms()
    assert method in list(methods.keys()), \
        'please provide correct method string'

    if method == 'siz':
        M = shadow_index_zhou(Blue, Green, Red)
    elif method == 'isi':
        M = improved_shadow_index(Blue, Green, Red, Near)
    elif method == 'sei':
        M = shadow_enhancement_index(Blue, Green, Red, Near)
    elif method == 'fcsdi':
        M = false_color_shadow_difference_index(Green, Red, Near)
    elif method == 'csi':
        M = combinational_shadow_index(Blue, Green, Red, Near)
    elif method == 'nsvdi':
        M = normalized_sat_value_difference_index(Blue, Green, Red)
    elif method == 'sil':
        M = shadow_index_liu(Blue, Green, Red)
    elif method == 'sr':
        M = specthem_ratio(Blue, Green, Red)
    elif method == 'sdi':
        M = shadow_detector_index(Blue, Green, Red)
    elif method == 'sisdi':
        M = sat_int_shadow_detector_index(Blue, RedEdge, Near)
    elif method == 'mixed':
        M = mixed_property_based_shadow_index(Blue, Green, Red)
    elif method == 'msf':
        p_s = kwargs['P_S'] if 'P_S' in kwargs else .95
        M = modified_shadow_fraction(Blue, Green, Red, P_S=p_s)
    elif method == 'c3':
        M = color_invariant(Blue, Green, Red)
    elif method == 'entropy':
        a = kwargs['a'] if 'a' in kwargs else 138.
        M, R = entropy_shade_removal(Green, Red, Near, a=a)
        return M, R
    elif method == 'shi':
        M = shade_index(Blue, Green, Red, Near)
    elif method == 'sp':
        ae = kwargs['ae'] if 'ae' in kwargs else 1e+1
        be = kwargs['be'] if 'be' in kwargs else 5e-1
        M = shadow_probabilities(Blue, Green, Red, Near, ae=ae, be=be)
    elif method in (
            'nri',
            'normalized_range_shadow_index',
    ):
        M = normalized_range_shadow_index(Blue, Green, Red, Near)
    else:
        M = None
    return M


# generic transforms


# generic metrics
def shannon_entropy(X, band_width=100):
    """ calculate shanon entropy from data sample

    Parameters
    ----------
    X : numpy.ndarray, size=(m,1)
        data array
    band_width : integer, optional
        sample bandwith

    Returns
    -------
    shannon : float
        entropy value of the total data sample
    """
    assert isinstance(X, np.ndarray), 'please provide an array'

    num_bins = np.round(np.nanmax(X) - np.nanmin(X) / band_width)
    hist, bin_edges = np.histogram(X.flatten(),
                                   bins=num_bins.astype(int),
                                   density=True)
    hist = hist[hist != 0]
    shannon = -np.sum(hist * np.log2(hist))
    return shannon


# color-based shadow functions
def shadow_index_zhou(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index, based upon [Zh21]_.

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See Also
    --------
    improved_shadow_index

    References
    ----------
    .. [Zh21] Zhou et al. "Shadow detection and compensation from remote
              sensing images under complex urban conditions", 2021.
    """
    are_three_arrays_equal(Blue, Green, Red)

    Y, Cb, Cr = rgb2ycbcr(Red, Green, Blue)
    SI = np.divide(Cb - Y, Cb + Y)
    return SI


def shadow_hsv_fraction(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index. Based upon [Ts06].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hcv_fraction, shadow_ycbcr_fraction, shadow_yiq_fraction,
    modified_shadow_fraction

    References
    ----------
    .. [Ts06] Tsai. "A comparative study on shadow compensation of color aerial
       images in invariant color models." IEEE Transactions on geoscience and
       remote sensing vol.44 pp.1661–1671, 2006.
    """
    are_three_arrays_equal(Blue, Green, Red)

    H, _, Int = rgb2hsi(Red, Green, Blue)
    SF = np.divide(H + 1, Int + 1, where=Int != -1)
    return SF


def modified_shadow_fraction(Blue, Green, Red, P_S=.95):
    """transform red, green, blue arrays to shadow index. Based upon [Ch09].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hsv_fraction

    References
    ----------
    .. [Ch09] Chung, et al. "Efficient shadow detection of color aerial images
       based on successive thresholding scheme." IEEE transactions on
       geoscience and remote sensing vol.47, pp.671–682, 2009.
    """
    are_three_arrays_equal(Blue, Green, Red)

    Hue, _, Int = rgb2hsi(Red, Green, Blue)
    r = np.divide(Hue, Int + 1, where=Int != -1)
    T_S = np.quantile(r, P_S)
    sig = np.std(r)

    SF = np.exp(-np.divide((r - T_S)**2, 4 * sig))
    np.putmask(SF, SF >= T_S, 1.)
    return SF


def shadow_hcv_fraction(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index. Based upon [Ts06].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hsv_fraction, shadow_ycbcr_fraction, shadow_yiq_fraction

    References
    ----------
    .. [Ts06] Tsai. "A comparative study on shadow compensation of color aerial
       images in invariant color models." IEEE Transactions on geoscience and
       remote sensing vol.44 pp.1661–1671, 2006.
    """
    are_three_arrays_equal(Blue, Green, Red)

    H, _, V = rgb2hcv(Red, Green, Blue)
    SF = np.divide(H + 1, V + 1, where=V != -1)
    return SF


def shadow_ycbcr_fraction(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index. Based upon [Ts06].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hcv_fraction, shadow_hsv_fraction, shadow_yiq_fraction

    References
    ----------
    .. [Ts06] Tsai. "A comparative study on shadow compensation of color aerial
       images in invariant color models." IEEE Transactions on geoscience and
       remote sensing vol.44 pp.1661–1671, 2006.
    """
    are_three_arrays_equal(Blue, Green, Red)

    Y, _, Cr = rgb2ycbcr(Red, Green, Blue)
    SF = np.divide(Cr + 1, Y + 1, where=Y != -1)
    return SF


def shadow_yiq_fraction(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index. Based upon [Ts06].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hcv_fraction, shadow_ycbcr_fraction

    References
    ----------
    .. [Ts06] Tsai. "A comparative study on shadow compensation of color aerial
       images in invariant color models." IEEE Transactions on geoscience and
       remote sensing vol.44 pp.1661–1671, 2006.
    """
    are_three_arrays_equal(Blue, Green, Red)

    Y, _, Q = rgb2yiq(Red, Green, Blue)
    SF = np.divide(Q + 1, Y + 1, where=Y != -1)
    return SF


# def shadow_quantifier_index
# Polidorio, A. M., Flores F. C., Imai N. N., Tommaselli, A. M. G. and Franco,
# C. 2003. Automatic Shadow Segmentation in Aerial Color Images. In:
# Proceedings of the XVI SIBGRAPI. XVI Brazilian Symposium on Computer Graphics
# and Image Processing. São Carlos, Brasil, 12-15 October 2003.
# doi:10.1109/SIBGRA.2003.1241019.
def improved_shadow_index(Blue, Green, Red, Near):
    """transform red, green, blue arrays to improved shadow index. Based upon
    [Zh21].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    ISI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_index_zhou

    Notes
    -----
    .. math:: ISI = (SI + (1-I_{near}))/(SI + (1+I_{near}))

    References
    ----------
    .. [Zh21] Zhou et al. "Shadow detection and compensation from remote
              sensing images under complex urban conditions", 2021.
    """
    are_three_arrays_equal(Blue, Green, Red)
    are_two_arrays_equal(Blue, Near)

    SI = shadow_index_zhou(Blue, Green, Red)
    ISI = np.divide(SI + (1 - Near), SI + (1 + Near))

    return ISI


def shadow_enhancement_index(Blue, Green, Red, Near):
    """ transform red, green, blue arrays to SEI index. See also [Su19].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    SEI : numpy.ndarray, size=(m,n)
        array with sei transform

    See Also
    --------
    combinational_shadow_index

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: SEI = [(B_{1}+B_{9})-(B_{3}+B_{8})]/[(B_{1}+B_{9})+(B_{3}+B_{8})]



    References
    ----------
    .. [Su19] Sun et al. "Combinational shadow index for building shadow
       extraction in urban areas from Sentinel-2A MSI imagery" International
       journal of applied earth observation and geoinformation, vol.78
       pp.53--65, 2019.
    """
    are_three_arrays_equal(Blue, Green, Red)
    are_two_arrays_equal(Blue, Near)

    denom = (Blue + Near) + (Green + Red)
    SEI = np.divide((Blue + Near) - (Green + Red), denom, where=denom != 0)
    return SEI


def false_color_shadow_difference_index(Green, Red, Near):
    """transform red and near infrared arrays to shadow index. See also [Te11].

    Parameters
    ----------
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    FCSI : numpy.ndarray, size=(m,n)
        array with shadow enhancement

    See also
    --------
    normalized_sat_value_difference_index, rgb2hsi

    Notes
    -----
    .. math:: FCSI = (I_{sat} - I_{int}) / (I_{sat} + I_{int})

    References
    ----------
    .. [Te11] Teke et al. "Multi-spectral false color shadow detection",
       Proceedings of the ISPRS conference on photogrammetric image analysis,
       pp.109-119, 2011.
    """
    are_three_arrays_equal(Green, Red, Near)

    _, Sat, Int = rgb2hsi(Near, Red, Green)  # create HSI bands
    denom = Sat + Int
    FCSI = np.divide(Sat - Int, denom, where=denom != 0)
    return FCSI


def normalized_difference_water_index(Green, Near):
    """transform green and near infrared arrays NDW-index. See also [Ga96].

    Parameters
    ----------
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    NDWI : numpy.ndarray, size=(m,n)
        array with NDWI transform

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: NDWI = [B_{3}-B_{8}]/[B_{3}+B_{8}]

    References
    ----------
    .. [Ga96] Gao, "NDWI - a normalized difference water index for remote
       sensing of vegetation liquid water from space" Remote sensing of
       environonment, vol.58 pp.257–266, 1996
    """
    are_two_arrays_equal(Green, Near)

    denom = (Green + Near)
    NDWI = np.divide((Green - Near),
                     denom,
                     out=np.zeros_like(denom),
                     where=denom != 0)
    return NDWI


def modified_normalized_difference_water_index(Green, Short):
    """transform green and shortwave infrared arrays to modified NDW-index.
    See also [Xu06]_.

    Parameters
    ----------
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Shortwave : numpy.ndarray, size=(m,n)
        shortwave infrared band of satellite image, for Sentinel-2 this is B11,
        originally the Landsat TM band 5 was used in [Xu06]_.

    Returns
    -------
    NDWI : numpy.ndarray, size=(m,n)
        array with NDWI transform

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: NDWI = [B_{3}-B_{11}]/[B_{3}+B_{11}]

    References
    ----------
    .. [Xu06] Xu, "Modification of normalized difference water index (NDWI) to
              enhance open water features in remotely sensed imagery"
              International journal of remote sensing, vol.27(14) pp.3025–3033,
              2006.
    """
    are_two_arrays_equal(Green, Short)

    denom = (Green + Short)
    MNDWI = np.divide((Green - Short),
                      denom,
                      out=np.zeros_like(denom),
                      where=denom != 0)
    return MNDWI


def normalized_difference_moisture_index(Near, Short):
    """transform near and shortwave infrared arrays to NDM-index.
    See also [Wi02]_.

    Parameters
    ----------
    Near : numpy.ndarray, size=(m,n)
        green band of satellite image
    Shortwave : numpy.ndarray, size=(m,n)
        shortwave infrared band of satellite image

    Returns
    -------
    NDMI : numpy.ndarray, size=(m,n)
        array with NDMI transform

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: NDMI = [B_{8}-B_{11}]/[B_{8}+B_{11}]

    References
    ----------
    .. [Wi02] Wilson, "Detection of forest harvest type using multiple dates of
              Landsat TM imagery" Remote sensing of environment, vol.80
              pp.385–396, 2002.
    """
    are_two_arrays_equal(Near, Short)

    denom = (Near + Short)
    NDMI = np.divide((Near - Short),
                     denom,
                     out=np.zeros_like(denom),
                     where=denom != 0)
    return NDMI


def normalized_difference_blue_water_index(Blue, Near):
    """transform green and near infrared arrays NDW-index. See also [Qu11].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    NDWI : numpy.ndarray, size=(m,n)
        array with NDWI transform

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: NDWI = [B_{3}-B_{8}]/[B_{3}+B_{8}]

    References
    ----------
    .. [Qu11] Qu et al, "Research on automatic extraction of water bodies and
       wetlands on HJU satellite CCD images" Remote sensing information,
       vol.4 pp.28–33, 2011
    """
    are_two_arrays_equal(Blue, Near)

    denom = (Blue + Near)
    NDBWI = np.divide((Blue - Near),
                      denom,
                      out=np.zeros_like(denom),
                      where=denom != 0)
    return NDBWI


def combinational_shadow_index(Blue, Green, Red, Near):
    """transform red, green, blue arrays to combined shadow index. See also
    [Su19].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    CSI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_enhancement_index

    References
    ----------
    .. [Su19] Sun et al. "Combinational shadow index for building shadow
       extraction in urban areas from Sentinel-2A MSI imagery" International
       journal of applied earth observation and geoinformation, vol.78
       pp.53--65, 2019.
    """
    are_three_arrays_equal(Blue, Green, Red)
    are_two_arrays_equal(Blue, Near)

    SEI = shadow_enhancement_index(Blue, Green, Red, Near)
    NDWI = normalized_difference_water_index(Green, Near)

    option_1 = SEI - Near
    option_2 = SEI - NDWI
    IN_1 = Near >= NDWI

    CSI = option_2
    CSI[IN_1] = option_1[IN_1]
    return CSI


def normalized_sat_value_difference_index(Blue, Green, Red):
    """ transform red, green, blue arrays to normalized sat-value difference.
    See also [Ma08].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    NSVDI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See also
    --------
    false_color_shadow_difference_index, rgb2hsi

    Notes
    -----
    .. math:: NSVDI = (I_{sat} - I_{int}) / (I_{sat} + I_{int})

    References
    ----------
    .. [Ma08] Ma et al. "Shadow segmentation and compensation in high
       resolution satellite images", Proceedings of IEEE IGARSS,
       pp.II-1036--II-1039, 2008.
    """
    are_three_arrays_equal(Blue, Green, Red)

    _, Sat, Int = rgb2hsi(Red, Green, Blue)  # create HSI bands
    denom = Sat + Int
    NSVDI = np.divide(Sat - Int, denom, where=denom != 0)
    return NSVDI


def shadow_identification(Blue, Green, Red):
    """ transform red, green, blue arrays to sat-value difference, following
    [Po03].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : numpy.ndarray, size=(m,n)
        array with shadow transform

    See also
    --------
    false_color_shadow_difference_index, rgb2hsi

    Notes
    -----
    .. math:: SI = I_{sat} - I_{int}

    References
    ----------
    .. [Po03] Polidorio et al. "Automatic shadow segmentation in aerial color
       images", Proceedings of the 16th Brazilian symposium on computer
       graphics and image processing, pp.270-277, 2003.
    """
    are_three_arrays_equal(Blue, Green, Red)

    _, Sat, Int = rgb2hsi(Red, Green, Blue)  # create HSI bands
    SI = Sat - Int
    return SI


def shadow_index_liu(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index, see [LX13] and
    [Su19].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : numpy.ndarray, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [LX13] Liu & Xie. "Study on shadow detection in high resolution remote
       sensing image of PCA and HIS model", Remote sensing technology and
       application, vol.28(1) pp. 78--84, 2013.
    .. [Su19] Sun et al. "Combinational shadow index for building shadow
       extraction in urban areas from Sentinel-2A MSI imagery" International
       journal of applied earth observation and geoinformation, vol.78
       pp.53--65, 2019.
    """
    are_three_arrays_equal(Blue, Green, Red)

    _, Sat, Int = rgb2hsi(Red, Green, Blue)

    X = pca_rgb_preparation(Red, Green, Blue, min_samp=1e4)
    e, lamb = principle_component_analysis(X)
    del X
    PC1 = e[0, 0] * Red + e[0, 1] * Green + e[0, 2] * Blue

    denom = (PC1 + Int + Sat)
    SI = np.divide((PC1 - Int) * (1 + Sat), denom, where=denom != 0)
    return SI


def specthem_ratio(Blue, Green, Red):
    """transform red, green, blue arrays to specthem ratio, see [Si18].

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image

    Returns
    -------
    Sr : numpy.ndarray, size=(m,n)
        array with shadow enhancement

    Notes
    -----
    In the paper the logarithmic is also used further to enhance the shadows,

    .. math:: Sr = log{[Sr+1]}

    References
    ----------
    .. [Si18] Silva et al. "Near real-time shadow detection and removal in
       aerial motion imagery application" ISPRS journal of photogrammetry and
       remote sensing, vol.140 pp.104--121, 2018.
    """
    are_three_arrays_equal(Blue, Green, Red)

    X, Y, Z = rgb2xyz(Red, Green, Blue, method='Ford')
    L, a, b = xyz2lab(X, Y, Z)
    h = lab2lch(L, a, b)[1]
    Sr = np.divide(h + 1, L + 1)
    return Sr


def shadow_detector_index(Blue, Green, Red):
    """transform red, green, blue arrays to shadow detector index, see [MA17].

    Parameters
    ----------
    Blue : numpy.array, size=(m,n)
        blue band of satellite image
    Green : numpy.array, size=(m,n)
        green band of satellite image
    Red : numpy.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SDI : numpy.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [MA17] Mustafa & Abedehafez. "Accurate shadow detection from
       high-resolution satellite images" IEEE geoscience and remote sensing
       letters, vol.14(4) pp.494--498, 2017.
    """
    are_three_arrays_equal(Blue, Green, Red)

    X = pca_rgb_preparation(Red, Green, Blue, min_samp=1e4)
    e, lamb = principle_component_analysis(X)
    del X
    PC1 = e[0, 0] * Red + e[0, 1] * Green + e[0, 2] * Blue

    denom = ((Green - Blue) * Red) + 1
    SDI = np.divide((1 - PC1) + 1, denom, where=denom != 0)
    return SDI


def sat_int_shadow_detector_index(Blue, RedEdge, Near):
    """ transform red, green, blue arrays to sat-int shadow detector index, see
    also [MA18].

    Parameters
    ----------
    Blue : numpy.array, size=(m,n)
        blue band of satellite image
    RedEdge : numpy.array, size=(m,n)
        rededge band of satellite image
    Near : numpy.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    SISDI : numpy.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [MA18] Mustafa & Abedelwahab. "Corresponding regions for shadow
       restoration in satellite high-resolution images" International journal
       of remote sensing, vol.39(20) pp.7014--7028, 2018.
    """
    are_three_arrays_equal(Blue, RedEdge, Near)

    _, Sat, Int = erdas2hsi(Blue, RedEdge, Near)
    SISDI = Sat - (2 * Int)
    return SISDI


def mixed_property_based_shadow_index(Blue, Green, Red):
    """ transform red, green, blue arrays to Mixed property-based shadow index,
    see also [Ha18].

    Parameters
    ----------
    Blue : numpy.array, size=(m,n)
        blue band of satellite image
    Green : numpy.array, size=(m,n)
        green band of satellite image
    Red : numpy.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    MPSI : numpy.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [Ha18] Han et al. "A mixed property-based automatic shadow detection
       approach for VHR multispectral remote sensing images" Applied sciences
       vol.8(10) pp.1883, 2018.
    """
    are_three_arrays_equal(Blue, Green, Red)

    Hue, _, Int = rgb2hsi(Red, Green, Blue)  # create HSI bands
    MPSI = np.multiply(Hue - Int, Green - Blue)
    return MPSI


def color_invariant(Blue, Green, Red):
    """ transform red, green, blue arrays to color invariant (c3), see also
    [GS99] and [AA08].

    Parameters
    ----------
    Blue : numpy.array, size=(m,n)
        blue band of satellite image
    Green : numpy.array, size=(m,n)
        green band of satellite image
    Red : numpy.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    c3 : numpy.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [GS99] Gevers & Smeulders. "Color-based object recognition" Pattern
       recognition, vol.32 pp.453--464, 1999
    .. [AA08] Arévalo González & Ambrosio. "Shadow detection in colour
       high‐resolution satellite images" International journal of remote
       sensing, vol.29(7) pp.1945--1963, 2008
    """
    are_three_arrays_equal(Blue, Green, Red)

    c3 = np.arctan2(Blue, np.amax(np.dstack((Red, Green))))
    return c3


def modified_color_invariant(Blue, Green, Red, Near):  # wip
    """ transform red, green, blue arrays to color invariant (c3), see also
    [GS99] and [BA15].

    Parameters
    ----------
    Blue : numpy.array, size=(m,n)
        blue band of satellite image
    Green : numpy.array, size=(m,n)
        green band of satellite image
    Red : numpy.array, size=(m,n)
        red band of satellite image
    Near : numpy.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    c3 : numpy.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [GS99] Gevers & Smeulders. "Color-based object recognition" Pattern
       recognition, vol.32 pp.453--464, 1999
    .. [BA15] Besheer & Abdelhafiz. "Modified invariant colour model for shadow
       detection" International journal of remote sensing, vol.36(24)
       pp.6214--6223, 2015.
    """
    are_three_arrays_equal(Blue, Green, Red)
    are_two_arrays_equal(Blue, Near)

    C3 = color_invariant(Blue, Green, Red)
    NDWI = normalized_difference_water_index(Green, Near)

    # [BA15] Uses thresholding and boolean algebra,
    # but here multiplication is used, so a float is constructed
    MCI = np.multiply(C3, NDWI)
    return MCI


def reinhard(Blue, Green, Red):
    """ transform red, green, blue arrays to luminance, see also [Re01].

    Parameters
    ----------
    Blue : numpy.array, size=(m,n)
        blue band of satellite image
    Green : numpy.array, size=(m,n)
        green band of satellite image
    Red : numpy.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    l : numpy.array, size=(m,n)
        shadow enhanced band

    References
    ----------
    .. [Re01] Reinhard et al. "Color transfer between images" IEEE Computer
       graphics and applications vol.21(5) pp.34-41, 2001.
    """
    are_three_arrays_equal(Blue, Green, Red)

    (L, M, S) = rgb2lms(Red, Green, Blue)
    L = np.log10(L)
    M = np.log10(M)
    S = np.log10(S)
    l, _, _ = lms2lab(L, M, S)
    return l


def entropy_shade_removal(Ia, Ib, Ic, a=None):
    """ Make use of Wiens' law to get illumination component, see also [Fi09].

    Parameters
    ----------
    Ia : numpy.array, size=(m,n)
        highest frequency band of satellite image
    Ib : numpy.array, size=(m,n)
        lower frequency band of satellite image
    Ic : numpy.array, size=(m,n)
        lowest frequency band of satellite image
    a : float, default=-1
        angle to use [in degrees]

    Returns
    -------
    S : numpy.array, size=(m,n)
        shadow enhanced band

    References
    ----------
    .. [Fi09] Finlayson et al. "Entropy minimization for shadow removal"
       International journal of computer vision vol.85(1) pp.35-57 2009.
    """
    are_three_arrays_equal(Ia, Ib, Ic)

    Ia[Ia == 0], Ib[Ib == 0] = 1E-6, 1E-6
    sqIc = np.power(Ic, 1. / 2)

    chi1 = np.log(np.divide(Ia, sqIc, out=np.zeros_like(Ic), where=sqIc != 0),
                  where=sqIc != 0)
    chi2 = np.log(np.divide(Ib, sqIc, out=np.zeros_like(Ic), where=sqIc != 0),
                  where=sqIc != 0)

    if isinstance(
            a, type(None)):  # estimate angle from data if angle is not given
        (m, n) = chi1.shape
        mn = np.power((m * n), -1. / 3)
        # loop through angles
        angl = np.arange(0, 180)
        shan = np.zeros(angl.shape)
        for i in angl:
            chi_rot = np.cos(np.radians(i)) * chi1 + \
                      np.sin(np.radians(i)) * chi2
            band_w = 3.5 * np.std(chi_rot) * mn
            shan[i] = shannon_entropy(chi_rot, band_w)

        a = angl[np.argmin(shan)]

    # create imagery
    S = np.cos(np.radians(a)) * chi1 + np.sin(np.radians(a)) * chi2
    b = a - 90
    # b = angl[np.argmax(shan)]
    R = np.cos(np.radians(b)) * chi1 + np.sin(np.radians(b)) * chi2
    return S, R


def shade_index(*args):
    """ transform multi-spectral arrays to shade index, see also [HS10] and
    [Al12].

    Parameters
    ----------
    *args : numpy.array, size=(m,n)
        different bands of a satellite image, all of the same extent

    Returns
    -------
    SI : numpy.array, size=(m,n)
        shadow enhanced band

    Notes
    -----
    Amplify dark regions by simple multiplication:

    .. math:: SI = Pi_{i}^{N}{[1 - B_{i}]}

    References
    ----------
    .. [HS10] Hogan & Smith, "Refinement of digital elvation models from
       shadowing cues" IEEE computer society conference on computer vision and
       pattern recognition, 2010.
    .. [Al12] Altena, "Filling the white gap on the map: Photoclinometry for
       glacier elevation modelling" MSc thesis TU Delft, 2012.
    """
    im_stack = np.stack(args, axis=2)
    SI = np.prod(1 - im_stack, axis=2)
    return SI


def normalized_range_shadow_index(*args):
    """transform multi-spectral arrays to shadow index

    Parameters
    ----------
    *args : numpy.array, size=(m,n)
        different bands of a satellite image, all of the same extent

    Returns
    -------
    SI : numpy.array, size=(m,n)
        shadow enhanced band

    Notes
    -----
    Schematically this looks like:

        .. code-block:: text

          I1    ┌-----┐
           -┬---┤     |
          I2|┌--┤     | Imin
           -┼┼┬-┤ min ├------┐
            |||┌┤     |      |┌-------┐
            ||||└-----┘      └┤(a-b)/ |   SI
          I3||||┌-----┐      ┌┤  (a+b)├------
           -┼┴┼┼┤     | Imax |└-------┘
          I4└-┼┼┤     ├------┘
          ----┼┴┤ max |
              └-┤     |
                └-----┘


    """

    im_stack = np.stack(args, axis=2)
    im_min, im_max = np.min(im_stack, axis=2), np.max(im_stack, axis=2)
    im_dif, im_rng = im_max - im_min, im_max + im_min
    SI = np.divide(im_dif,
                   im_rng,
                   out=np.zeros_like(im_rng),
                   where=np.abs(im_rng) != 0)
    return SI


def fractional_range_shadow_index(*args):
    """transform multi-spectral arrays to shadow index

    Parameters
    ----------
    *args : numpy.array, size=(m,n)
        different bands of a satellite image, all of the same extent

    Returns
    -------
    SI : numpy.array, size=(m,n)
        shadow enhanced band

    Notes
    -----
    Schematically this looks like:

        .. code-block:: text

          I1    ┌-----┐
           -┬---┤     |
          I2|┌--┤     | Imin
           -┼┼┬-┤ min ├------┐
            |||┌┤     |      |┌-------┐
            ||||└-----┘      └┤ a / b |   SI
          I3||||┌-----┐      ┌┤       ├------
           -┼┴┼┼┤     | Imax |└-------┘
          I4└-┼┼┤     ├------┘
          ----┼┴┤ max |
              └-┤     |
                └-----┘
    """

    im_stack = np.stack(args, axis=2)
    im_min, im_max = np.min(im_stack, axis=2), np.max(im_stack, axis=2)
    SI = np.divide(im_min,
                   im_max,
                   out=np.zeros_like(im_max),
                   where=np.abs(im_max) != 0)
    return SI


def shadow_probabilities(Blue, Green, Red, Near, ae=1e+1, be=5e-1):
    """transform blue, green, red and near infrared to shade probabilities, see
    also [FS10] and [Rü13].

    Parameters
    ----------
    Blue : numpy.array, size=(m,n)
        blue band of satellite image
    Green : numpy.array, size=(m,n)
        green band of satellite image
    Red : numpy.array, size=(m,n)
        red band of satellite image
    Near : numpy.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    M : numpy.array, size=(m,n)
        shadow enhanced band

    References
    ----------
    .. [FS10] Fredembach & Süsstrunk. "Automatic and accurate shadow detection
       from (potentially) a single image using near-infrared information" EPFL
       Tech Report 165527, 2010.
    .. [Rü13] Rüfenacht et al. "Automatic and accurate shadow detection using
       near-infrared information" IEEE transactions on pattern analysis and
       machine intelligence, vol.36(8) pp.1672-1678, 2013.
    """
    are_three_arrays_equal(Blue, Green, Red)
    are_two_arrays_equal(Blue, Near)

    Fk = np.amax(np.stack((Red, Green, Blue), axis=2), axis=2)
    F = np.divide(np.clip(Fk, 0, 2), 2)  # (10) in [FS10]
    L = np.divide(Red + Green + Blue, 3)  # (4) in [FS10]
    del Red, Green, Blue, Fk

    Dvis = s_curve(1 - L, ae, be)
    Dnir = s_curve(1 - Near, ae, be)  # (5) in [FS10]
    D = np.multiply(Dvis, Dnir)  # (6) in [FS10]
    del L, Near, Dvis, Dnir

    M = np.multiply(D, (1 - F))
    return M


# recovery - normalized color composite

# todo: Makaru 2011, for reflectance values (L2 data)
# Wu 2007, Natural Shadow Matting. DOI: 10.1145/1243980.1243982
# https://github.com/yaksoy/AffinityBasedMattingToolbox
