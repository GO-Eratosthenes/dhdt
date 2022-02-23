import numpy as np
import random

from scipy import ndimage # for image filters
from sklearn.decomposition import fastica
from skimage.color import rgb2hsv, hsv2rgb

from .color_transforms import \
    rgb2ycbcr, rgb2hsi, rgb2xyz, xyz2lab, lab2lch, erdas2hsi, rgb2lms, lms2lab,\
    hsi2rgb, rgb2hcv, rgb2yiq
from .image_transforms import s_curve

def apply_shadow_transform(method, Blue, Green, Red, RedEdge, Near, Shw,
                           **kwargs):
    """ Given a specific method, employ shadow transform

    Parameters
    ----------
    method : {‘siz’,’ isi’,’ sei’,’ fcsdi’, ‘nsvi’, ‘nsvdi’, ‘sil’, ‘sr’,\
              ‘sdi’, ‘sisdi’, ‘mixed’, ‘c3’, ‘entropy’, ‘shi’,’ sp’}
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
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    RedEdge : np.array, size=(m,n)
        red edge band of satellite image
    Near : np.array, size=(m,n)
        near-infrared band of satellite image
    Shw : np.array, size=(m,n)
        synthetic shading from for example a digital elevation model

    Returns
    -------
    M : np.array, size=(m,n)
        shadow enhanced satellite image

    See Also
    --------
    .handler_multispec.get_shadow_bands :
        provides Blue, Green, Red bandnumbers for a specific satellite
    .handler_multispec.read_shadow_bands :
        reads the specific bands given
    """
    method = method.lower()

    if method=='siz':
        M = shadow_index_zhou(Blue, Green, Red)
    elif method=='isi':
        M = improved_shadow_index(Blue, Green, Red, Near)
    elif method=='sei':
        M = shadow_enhancement_index(Blue,Green,Red,Near)
    elif method=='fcsdi':
        M = false_color_shadow_difference_index(Green,Red,Near)
    elif method=='csi':
        M = combinational_shadow_index(Blue,Green,Red,Near)
    elif method=='nsvdi':
        M = normalized_sat_value_difference_index(Blue, Green, Red)
    elif method=='sil':
        M = shadow_index_liu(Blue, Green, Red)
    elif method=='sr':
        M = specthem_ratio(Blue, Green, Red)
    elif method=='sdi':
        M = shadow_detector_index(Blue, Green, Red)
    elif method=='sisdi':
        M = sat_int_shadow_detector_index(Blue, RedEdge, Near)
    elif method=='mixed':
        M = mixed_property_based_shadow_index(Blue, Green, Red)
    elif method=='msf':
        p_s = kwargs['P_S'] if kwargs['P_S'] is not None else .95
        M = modified_shadow_fraction(Blue, Green, Red, P_S=p_s)
    elif method=='c3':
        M = color_invariant(Blue, Green, Red)
    elif method=='entropy':
        a = kwargs['a'] if kwargs['a'] is not None else None
        M,R = entropy_shade_removal(Green, Red, Near, a=a)
        return M,R
    elif method=='shi':
        M = shade_index(Blue, Green, Red, Near)
    elif method=='sp':
        ae = kwargs['ae'] if kwargs['ae'] is not None else 1e+1
        be = kwargs['be'] if kwargs['be'] is not None else 5e-1
        M = shadow_probabilities(Blue, Green, Red, Near, ae=ae, be=be)

    return M

# generic transforms
def pca(X):
    """ principle component analysis (PCA)

    Parameters
    ----------
    X : np.array, size=(m,b)
        array of the band image

    Returns
    -------
    eigen_vecs : np.array, size=(b,b)
        array with eigenvectors
    eigen_vals : np.array, size=(b,1)
        vector with eigenvalues

    """
    assert type(X)==np.ndarray, ('please provide an array')

    # Data matrix X, assumes 0-centered
    n, m = X.shape
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n - 1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    return eigen_vecs, eigen_vals

def pca_rgb_preparation(Blue, Green, Red, min_samp=1e4):
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    OK = Blue != 0  # boolean array with data

    r,g,b = np.mean(Red[OK]), np.mean(Green[OK]), np.mean(Blue[OK])
    # shift axis origin
    Red,Green,Blue = Red-r, Green-g, Blue-b
    del r, g, b

    # sampleset
    Nsamp = int(
        min(min_samp, np.sum(OK)))  # try to have a sampleset of 10'000 points
    sampSet, other = np.where(OK)  # OK.nonzero()
    sampling = random.choices(sampSet, k=Nsamp)
    del sampSet, Nsamp

    Red = Red.ravel(order='F')
    Green = Green.ravel(order='F')
    Blue = Blue.ravel(order='F')

    X = np.transpose(np.array(
        [Red[sampling], Green[sampling], Blue[sampling]]))
    return X

def ica(Blue, Green, Red, Near, min_samp=1e4):
    OK = Blue != 0
    # sampleset
    Nsamp = int(
        min(min_samp, np.sum(OK)))  # try to have a sampleset of 10'000 points
    sampSet, other = np.where(OK)  # OK.nonzero()
    sampling = random.choices(sampSet, k=Nsamp)
    del sampSet, Nsamp
    X = np.array([Blue.flatten()[sampling],
                  Green.flatten()[sampling],
                  Red.flatten()[sampling],
                  Near.flatten()[sampling]]).T
    K,W,S,X_mean = fastica(X, return_X_mean=True)

    # reconstruct components
    first = W[0,0]*(Blue-X_mean[0])+W[0,1]*(Green-X_mean[1])+W[0,2]*(Red-X_mean[2])+W[0,3]*(Near-X_mean[3])
    secnd = W[1,0]*(Blue-X_mean[0])+W[1,1]*(Green-X_mean[1])+W[1,2]*(Red-X_mean[2])+W[1,3]*(Near-X_mean[3])
    third = W[2,0]*(Blue-X_mean[0])+W[2,1]*(Green-X_mean[1])+W[2,2]*(Red-X_mean[2])+W[2,3]*(Near-X_mean[3])
    forth = W[3,0]*(Blue-X_mean[0])+W[3,1]*(Green-X_mean[1])+W[3,2]*(Red-X_mean[2])+W[3,3]*(Near-X_mean[3])

    return first, secnd, third, forth

# generic metrics
def shannon_entropy(X,band_width=100):
    """ calculate shanon entropy from data sample

    Parameters
    ----------
    X : np.array, size=(m,1)
        data array
    band_width : integer, optional
        sample bandwith

    Returns
    -------
    shannon : float
        entropy value of the total data sample
    """
    assert type(X)==np.ndarray, ('please provide an array')

    num_bins = np.round(np.nanmax(X)-np.nanmin(X)/band_width)
    hist, bin_edges = np.histogram(X.flatten(),
                                   bins=num_bins.astype(int),
                                   density=True)
    hist = hist[hist!=0]
    shannon = -np.sum(hist * np.log2(hist))
    return shannon

# color-based shadow functions
def shadow_index_zhou(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : np.array, size=(m,n)
        array with shadow transform

    See Also
    --------
    improved_shadow_index

    References
    ----------
    .. [1] Zhou et al. "Shadow detection and compensation from remote sensing
           images under complex urban conditions", 2021.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    Y, Cb, Cr = rgb2ycbcr(Red, Green, Blue)
    SI = np.divide(Cb-Y, Cb+Y)
    return SI

def shadow_hsv_fraction(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : np.array, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hcv_fraction, shadow_ycbcr_fraction, shadow_yiq_fraction,
    modified_shadow_fraction

    References
    ----------
    .. [1] Tsai. "A comparative study on shadow compensation of color aerial
       images in invariant color models." IEEE Transactions on geoscience and
       remote sensing vol.44 pp.1661–1671, 2006.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    H, S, I = rgb2hsi(Red, Green, Blue)
    SF = np.divide(H+1, I+1)
    return SF

def modified_shadow_fraction(Blue, Green, Red, P_S=.95):
    """transform red, green, blue arrays to shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : np.array, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hsv_fraction

    References
    ----------
    .. [1] Chung, et al. "Efficient shadow detection of color aerial images
       based on successive thresholding scheme." IEEE transactions on geoscience
       and remote sensing vol.47, pp.671–682, 2009.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    H, S, I = rgb2hsi(Red, Green, Blue)
    r = np.divide(H, I+1)
    T_S = np.quantile(r, P_S)
    sig = np.std(r)

    SF = np.exp(-np.divide((r-T_S)**2, 4*sig))
    SF[SF>=T_S] = 1
    return SF

def shadow_hcv_fraction(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : np.array, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hsv_fraction, shadow_ycbcr_fraction, shadow_yiq_fraction

    References
    ----------
    .. [1] Tsai. "A comparative study on shadow compensation of color aerial
       images in invariant color models." IEEE Transactions on geoscience and
       remote sensing vol.44 pp.1661–1671, 2006.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    H, C, V = rgb2hcv(Red, Green, Blue)
    SF = np.divide(H+1, V+1)
    return SF

def shadow_ycbcr_fraction(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : np.array, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hcv_fraction, shadow_hsv_fraction, shadow_yiq_fraction

    References
    ----------
    .. [1] Tsai. "A comparative study on shadow compensation of color aerial
       images in invariant color models." IEEE Transactions on geoscience and
       remote sensing vol.44 pp.1661–1671, 2006.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    Y, Cb, Cr = rgb2ycbcr(Red, Green, Blue)
    SF = np.divide(Cr+1, Y+1)
    return SF

def shadow_yiq_fraction(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : np.array, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_hcv_fraction, shadow_ycbcr_fraction

    References
    ----------
    .. [1] Tsai. "A comparative study on shadow compensation of color aerial
       images in invariant color models." IEEE Transactions on geoscience and
       remote sensing vol.44 pp.1661–1671, 2006.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    Y, I, Q = rgb2yiq(Red, Green, Blue)
    SF = np.divide(Q+1, Y+1)
    return SF

#def shadow_quantifier_index
# Polidorio, A. M., Flores F. C., Imai N. N., Tommaselli, A. M. G. and Franco, C. 2003. Automatic Shadow Segmentation in Aerial Color Images. In: Proceedings of the XVI SIBGRAPI. XVI Brazilian Symposium on Computer Graphics and Image Processing. São Carlos, Brasil, 12-15 October 2003. doi:10.1109/SIBGRA.2003.1241019.
def improved_shadow_index(Blue, Green, Red, Near):
    """transform red, green, blue arrays to improved shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    ISI : np.array, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_index_zhou

    Notes
    -----
    .. math:: ISI = (SI + (1-I_{near}))/(SI + (1+I_{near}))

    References
    ----------
    .. [1] Zhou et al. "Shadow detection and compensation from remote sensing
       images under complex urban conditions", 2021.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    SI = shadow_index_zhou(Blue, Green, Red)
    ISI = np.divide( SI + (1 - Near), SI + (1 + Near))

    return ISI

def shadow_enhancement_index(Blue,Green,Red,Near):
    """transform red, green, blue arrays to sei? index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    SEI : np.array, size=(m,n)
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
    .. [1] Sun et al. "Combinational shadow index for building shadow
       extraction in urban areas from Sentinel-2A MSI imagery" International
       journal of applied earth observation and geoinformation, vol.78
       pp.53--65, 2019.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert type(Near)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')
    assert Blue.shape[:2] == Near.shape[:2], ('arrays should be of equal size')

    SEI = np.divide( (Blue + Near)-(Green + Red), (Blue + Near)+(Green + Red))
    return SEI

def false_color_shadow_difference_index(Green,Red,Near):
    """transform red and near infrared arrays to shadow index

    Parameters
    ----------
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    FCSI : np.array, size=(m,n)
        array with shadow enhancement

    See also
    --------
    normalized_sat_value_difference_index, rgb2hsi

    Notes
    -----
    .. math:: FCSI = (I_{sat} - I_{int}) / (I_{sat} + I_{int})

    References
    ----------
    .. [1] Teke et al. "Multi-spectral false color shadow detection",
       Proceedings of the ISPRS conference on photogrammetric image analysis,
       pp.109-119, 2011.
    """
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert type(Near)==np.ndarray, ('please provide an array')
    assert Red.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Red.shape[:2] == Near.shape[:2], ('arrays should be of equal size')

    (H, S, I) = rgb2hsi(Near, Red, Green)  # create HSI bands
    FCSI = (S - I) / (S + I)
    return FCSI

def normalized_difference_water_index(Green, Near):
    """transform green and near infrared arrays NDW-index

    Parameters
    ----------
    Green : np.array, size=(m,n)
        green band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    NDWI : np.array, size=(m,n)
        array with NDWI transform

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: NDWI = [B_{3}-B_{8}]/[B_{3}+B_{8}]

    References
    ----------
    .. [1] Gao, "NDWI - a normalized difference water index for remote sensing
       of vegetation liquid water from space" Remote sensing of environonment,
       vol. 58 pp. 257–266, 1996
    """
    assert type(Near)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert Green.shape[:2] == Near.shape[:2],('arrays should be of equal size')
    denom = (Green + Near)
    NDWI = np.divide( (Green - Near), denom,
                      out=np.zeros_like(denom), where=denom!=0)
    return NDWI

def normalized_difference_blue_water_index(Blue, Near):
    """transform green and near infrared arrays NDW-index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    NDWI : np.array, size=(m,n)
        array with NDWI transform

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: NDWI = [B_{3}-B_{8}]/[B_{3}+B_{8}]

    References
    ----------
    .. [1] Qu et al, "Research on automatic extraction of water bodies and
       wetlands on HJU satellite CCD images" Remote sensing information,
       vol.4 pp.28–33, 2011
    """
    assert type(Near)==np.ndarray, ('please provide an array')
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Near.shape[:2],('arrays should be of equal size')

    denom = (Blue + Near)
    NDBWI = np.divide( (Blue - Near), denom,
                      out=np.zeros_like(denom), where=denom!=0)
    return NDBWI

def combinational_shadow_index(Blue,Green,Red,Near):
    """transform red, green, blue arrays to combined shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    CSI : np.array, size=(m,n)
        array with shadow transform

    See Also
    --------
    shadow_enhancement_index

    References
    ----------
    .. [1] Sun et al. "Combinational shadow index for building shadow
       extraction in urban areas from Sentinel-2A MSI imagery" International
       journal of applied earth observation and geoinformation, vol.78
       pp.53--65, 2019.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert type(Near)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')
    assert Blue.shape[:2] == Near.shape[:2],('arrays should be of equal size')

    SEI = shadow_enhancement_index(Blue,Green,Red,Near)
    NDWI = normalized_difference_water_index(Green, Near)

    option_1 = SEI-Near
    option_2 = SEI-NDWI
    IN_1 = Near >= NDWI

    CSI = option_2
    CSI[IN_1] = option_1[IN_1]
    return CSI

def normalized_sat_value_difference_index(Blue, Green, Red):
    """transform red, green, blue arrays to normalized sat-value difference

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    NSVDI : np.array, size=(m,n)
        array with shadow transform

    See also
    --------
    false_color_shadow_difference_index, rgb2hsi

    Notes
    -----
    .. math:: NSVDI = (I_{sat} - I_{int}) / (I_{sat} + I_{int})

    References
    ----------
    .. [1] Ma et al. "Shadow segmentation and compensation in high resolution
       satellite images", Proceedings of IEEE IGARSS, pp.II-1036--II-1039,
       2008.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    (H,S,I) = rgb2hsi(Red, Green, Blue)  # create HSI bands
    NSVDI = (S - I) / (S + I)
    return NSVDI

def shadow_identification(Blue, Green, Red):
    """transform red, green, blue arrays to sat-value difference

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : np.array, size=(m,n)
        array with shadow transform

    See also
    --------
    false_color_shadow_difference_index, rgb2hsi

    Notes
    -----
    .. math:: SI = I_{sat} - I_{int}

    References
    ----------
    .. [1] Polidorio et al. "Automatic shadow segmentation in aerial color
       images", Proceedings of the 16th Brazilian symposium on computer graphics
       and image processing, pp.270-277, 2003.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    (H,S,I) = rgb2hsi(Red, Green, Blue)  # create HSI bands
    SI = S - I
    return SI

def shadow_index_liu(Blue, Green, Red):
    """transform red, green, blue arrays to shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SI : np.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [1] Liu & Xie. "Study on shadow detection in high resolution remote
           sensing image of PCA and HIS model", Remote sensing technology and
           application, vol.28(1) pp. 78--84, 2013.
    .. [2] Sun et al. "Combinational shadow index for building shadow
           extraction in urban areas from Sentinel-2A MSI imagery"
           International journal of applied earth observation and
           geoinformation, vol.78 pp.53--65, 2019.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    (_,S,I) = rgb2hsi(Red, Green, Blue)

    X = pca_rgb_preparation(Red, Green, Blue, min_samp=1e4)
    e, lamb = pca(X)
    del X
    PC1 = np.dot(e[0, :], np.array([Red, Green, Blue]))

    SI = np.divide((PC1 - I) * (1 + S), (PC1 + I + S))
    return SI

def specthem_ratio(Blue, Green, Red):
    """transform red, green, blue arrays to specthem ratio

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    Sr : np.array, size=(m,n)
        array with shadow enhancement

    Notes
    -----
    In the paper the logarithmic is also used further enhance the shadows,

    .. math:: Sr = log{[Sr+1]}

    References
    ----------
    .. [1] Silva et al. "Near real-time shadow detection and removal in aerial
       motion imagery application" ISPRS journal of photogrammetry and remote
       sensing, vol.140 pp.104--121, 2018.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    X,Y,Z = rgb2xyz(Red, Green, Blue, method='Ford')
    L,a,b = xyz2lab(X,Y,Z)
    _,h = lab2lch(L,a,b)
    Sr = np.divide(h+1 , L+1)
    return Sr

def shadow_detector_index(Blue, Green, Red):
    """transform red, green, blue arrays to shadow detector index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    SDI : np.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [1] Mustafa & Abedehafez. "Accurate shadow detection from
       high-resolution satellite images" IEEE geoscience and remote sensing
       letters, vol.14(4) pp.494--498, 2017.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    X = pca_rgb_preparation(Red, Green, Blue, min_samp=1e4)
    e, lamb = pca(X)
    del X
    PC1 = np.dot(e[0, :], np.array([Red, Green, Blue]))

    SDI = ((1 - PC1) + 1)/(((Green - Blue) * Red) + 1)
    return SDI

def sat_int_shadow_detector_index(Blue, RedEdge, Near):
    """transform red, green, blue arrays to sat-int shadow detector index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    RedEdge : np.array, size=(m,n)
        rededge band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    SISDI : np.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [1] Mustafa & Abedelwahab. "Corresponding regions for shadow restoration
       in satellite high-resolution images" International journal of remote
       sensing, vol.39(20) pp.7014--7028, 2018.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(RedEdge)==np.ndarray, ('please provide an array')
    assert type(Near)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == RedEdge.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Near.shape[:2], ('arrays should be of equal size')

    (_,S,I) = erdas2hsi(Blue, RedEdge, Near)
    SISDI = S - (2*I)
    return SISDI

def mixed_property_based_shadow_index(Blue, Green, Red):
    """transform Red Green Blue arrays to Mixed property-based shadow index

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    MPSI : np.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [1] Han et al. "A mixed property-based automatic shadow detection
       approach for VHR multispectral remote sensing images" Applied sciences
       vol.8(10) pp.1883, 2018.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    (H, S, I) = rgb2hsi(Red, Green, Blue)  # create HSI bands
    MPSI = (H - I) * (Green - Blue)
    return MPSI

def color_invariant(Blue, Green, Red):
    """transform Red Green Blue arrays to color invariant (c3)

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    c3 : np.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [1] Gevers & Smeulders. "Color-based object recognition" Pattern
       recognition, vol.32 pp.453--464, 1999
    .. [2] Arévalo González & G. Ambrosio. "Shadow detection in colour
       high‐resolution satellite images" International journal of remote
       sensing, vol.29(7) pp.1945--1963, 2008
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    c3 = np.arctan2( Blue, np.amax( np.dstack((Red, Green))))
    return c3

def modified_color_invariant(Blue, Green, Red, Near): #wip
    """transform Red Green Blue arrays to color invariant (c3)

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    c3 : np.array, size=(m,n)
        array with shadow enhancement

    References
    ----------
    .. [1] Gevers & Smeulders. "Color-based object recognition" Pattern
       recognition, vol.32 pp.453--464, 1999
    .. [2] Besheer & G. Abdelhafiz. "Modified invariant colour model for shadow
       detection" International journal of remote sensing, vol.36(24)
       pp.6214--6223, 2015.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert type(Near)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')
    assert Blue.shape[:2] == Near.shape[:2],('arrays should be of equal size')

    C3 = color_invariant(Blue, Green, Red)
    NDWI = normalized_difference_water_index(Green, Near)

    # [2] Uses thresholding and boolean algebra,
    # but here multiplication is used, so a float is constructed
    MCI = np.multiply(C3, NDWI)
    return MCI

def reinhard(Blue, Green, Red):
    """transform Red Green Blue arrays to luminance

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image

    Returns
    -------
    l : np.array, size=(m,n)
        shadow enhanced band

    References
    ----------
    .. [1] Reinhard et al. "Color transfer between images" IEEE Computer
       graphics and applications vol.21(5) pp.34-41, 2001.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')

    (L,M,S) = rgb2lms(Red, Green, Blue)
    L = np.log10(L)
    M = np.log10(M)
    S = np.log10(S)
    l,_,_ = lms2lab(L, M, S)
    return l

def entropy_shade_removal(Ia, Ib, Ic, a=None): #todo
    """

    Parameters
    ----------
    Ia : np.array, size=(m,n)
        highest frequency band of satellite image
    Ib : np.array, size=(m,n)
        lower frequency band of satellite image
    Ic : np.array, size=(m,n)
        lowest frequency band of satellite image
    a : float, default=-1
        angle to use [in degrees]

    Returns
    -------
    S : np.array, size=(m,n)
        shadow enhanced band

    References
    ----------
    .. [1] Finlayson et al. "Entropy minimization for shadow removal"
       International journal of computer vision vol.85(1) pp.35-57 2009.
    """
    assert type(Ia)==np.ndarray, ('please provide an array')
    assert type(Ib)==np.ndarray, ('please provide an array')
    assert type(Ic)==np.ndarray, ('please provide an array')
    assert Ia.shape[:2] == Ib.shape[:2],('arrays should be of equal size')
    assert Ia.shape[:2] == Ic.shape[:2], ('arrays should be of equal size')

    Ia[Ia==0] = 1e-6
    Ib[Ib==0] = 1e-6
    sqIc = np.power(Ic, 1./2)

    chi1 = np.log(np.divide( Ia, sqIc, out=np.zeros_like(Ic), where=sqIc!=0),
                  where=sqIc!=0)
    chi2 = np.log(np.divide( Ib, sqIc, out=np.zeros_like(Ic), where=sqIc!=0),
                  where=sqIc!=0)

    if isinstance(a, type(None)): # estimate angle from data if angle is not given
        (m,n) = chi1.shape
        mn = np.power((m*n),-1./3)
        # loop through angles
        angl = np.arange(0,180)
        shan = np.zeros(angl.shape)
        for i in angl:
            chi_rot = np.cos(np.radians(i))*chi1 + \
                np.sin(np.radians(i))*chi2
            band_w = 3.5*np.std(chi_rot)*mn
            shan[i] = shannon_entropy(chi_rot, band_w)

        a = angl[np.argmin(shan)]

    # create imagery
    S = np.cos(np.radians(a))*chi1 + np.sin(np.radians(a))*chi2
    b = a - 90
    #b = angl[np.argmax(shan)]
    R = np.cos(np.radians(b))*chi1 + np.sin(np.radians(b))*chi2
    return S,R

def shade_index(*args):
    """transform multi-spectral arrays to shade index

    Parameters
    ----------
    *args : np.array, size=(m,n)
        different bands of a satellite image, all of the same extent

    Returns
    -------
    SI : np.array, size=(m,n)
        shadow enhanced band

    Notes
    -----
    Amplify dark regions by simple multiplication:

    .. math:: SI = Pi_{i}^{N}{[1 - B_{i}]}

    References
    ----------
    .. [1] Altena, "Filling the white gap on the map: Photoclinometry for
       glacier elevation modelling" MSc thesis TU Delft, 2012.
    """
    im_stack = np.stack(args, axis=2)
    SI = np.prod(1 - im_stack, axis=2)
    return SI

def normalized_range_shadow_index(*args):
    """transform multi-spectral arrays to shadow index

    Parameters
    ----------
    *args : np.array, size=(m,n)
        different bands of a satellite image, all of the same extent

    Returns
    -------
    SI : np.array, size=(m,n)
        shadow enhanced band

    """

    im_stack = np.stack(args, axis=2)
    im_min, im_max = np.min(im_stack, axis=2), np.max(im_stack, axis=2)
    im_dif, im_rng = im_max-im_min, im_max+im_min
    SI = np.divide(im_dif, im_rng,
                   out=np.zeros_like(im_rng), where=np.abs(im_rng) != 0)
    return SI

def fractional_range_shadow_index(*args):
    """transform multi-spectral arrays to shadow index

    Parameters
    ----------
    *args : np.array, size=(m,n)
        different bands of a satellite image, all of the same extent

    Returns
    -------
    SI : np.array, size=(m,n)
        shadow enhanced band

    """

    im_stack = np.stack(args, axis=2)
    im_min, im_max = np.min(im_stack, axis=2), np.max(im_stack, axis=2)
    SI = np.divide(im_min, im_max,
                   out=np.zeros_like(im_max), where=np.abs(im_max) != 0)
    return SI

def shadow_probabilities(Blue, Green, Red, Near, ae = 1e+1, be = 5e-1):
    """transform blue, green, red and near infrared to shade probabilities

    Parameters
    ----------
    Blue : np.array, size=(m,n)
        blue band of satellite image
    Green : np.array, size=(m,n)
        green band of satellite image
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    M : np.array, size=(m,n)
        shadow enhanced band

    References
    ----------
    .. [1] Fredembach & Süsstrunk. "Automatic and accurate shadow detection
       from (potentially) a single image using near-infrared information" EPFL
       Tech Report 165527, 2010.
    .. [2] Rüfenacht et al. "Automatic and accurate shadow detection using
       near-infrared information" IEEE transactions on pattern analysis and
       machine intelligence, vol.36(8) pp.1672-1678, 2013.
    """
    assert type(Blue)==np.ndarray, ('please provide an array')
    assert type(Green)==np.ndarray, ('please provide an array')
    assert type(Red)==np.ndarray, ('please provide an array')
    assert type(Near)==np.ndarray, ('please provide an array')
    assert Blue.shape[:2] == Green.shape[:2],('arrays should be of equal size')
    assert Blue.shape[:2] == Red.shape[:2], ('arrays should be of equal size')
    assert Blue.shape[:2] == Near.shape[:2],('arrays should be of equal size')

    Fk = np.amax(np.stack((Red, Green, Blue), axis=2), axis=2)
    F = np.divide(np.clip(Fk, 0, 2), 2)  # (10), see Fredembach 2010
    L = np.divide(Red + Green + Blue, 3)  # (4), see Fredembach 2010
    del Red, Green, Blue, Fk

    Dvis = s_curve(1 - L, ae, be)
    Dnir = s_curve(1 - Near, ae, be)  # (5), see Fredembach 2010
    D = np.multiply(Dvis, Dnir)  # (6), see Fredembach 2010
    del L, Near, Dvis, Dnir

    M = np.multiply(D, (1 - F))
    return M

# recovery - normalized color composite

# todo: Makaru 2011, for reflectance values (L2 data)
# Wu 2007, Natural Shadow Matting. DOI: 10.1145/1243980.1243982
# https://github.com/yaksoy/AffinityBasedMattingToolbox
