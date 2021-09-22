import numpy as np
import random

from scipy import ndimage # for image filters

from .color_transforms import \
    rgb2ycbcr, rgb2hsi, rgb2xyz, xyz2lab, lab2lch, erdas2hsi, rgb2lms, lms2lab
from .image_transforms import s_curve

def enhance_shadow(method, Blue, Green, Red, RedEdge, Nir, Shw):
    """
    Given a specific method, employ shadow transform
    input:   method         string            method name to be implemented, 
                                              can be one of the following:
                                              - ruffenacht
                                              - nsvi
                                              - sdi
                                              - sisdi
                                              - mpsi
                                              - shadow_index
                                              - shade_index
                                              - levin
             Blue           array (n x m)     blue band of satellite image
             Green          array (n x m)     green band of satellite image
             Red            array (n x m)     red band of satellite image
             Nir            array (n x m)     near-infrared band of satellite image   
             Shw            array (n x m)     synthetic shading from for example
                                              a digital elevation model
    output:  M              array (n x m)     shadow enhanced satellite image
    """
    
    method = method.lower()
    
    if method=='ruffenacht':
        M = ruffenacht(Blue, Green, Red, Nir)
    elif len([n for n in ['shadow','index'] if n in method])==2:
        M = shadow_index(Blue, Green, Red, Nir)
    elif len([n for n in ['shade','index'] if n in method])==2:
        M = shade_index(Blue, Green, Red, Nir)
    elif method=='nsvi':
        M = nsvi(Blue, Green, Red)
    elif method=='sdi':
        M = sdi(Blue, Green, Red)
    elif method=='sisdi':
        M = sisdi(Blue, RedEdge, Nir)
    elif method=='mpsi':
        M = mpsi(Blue, Green, Red)    
    elif method=='gevers':
        M = gevers(Blue, Green, Red) 
    elif method=='finlayson':
        M = finlayson(Blue, Green, Red)
        
    return M
# generic transforms
def pca(X):
    """ principle component analysis (PCA)
    
    input:   data           array (n x m)     array of the band image
    output:  eigen_vecs     array (m x m)     array with eigenvectors
             eigen_vals     array (m x 1)     vector with eigenvalues
    """
    # Data matrix X, assumes 0-centered
    n, m = X.shape
    # removed, because a sample is taken:
    #    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n - 1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    return eigen_vecs, eigen_vals

def pca_rgb_preparation(Red, Green, Blue, min_samp=1e4):
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

# generic metrics
def shannon_entropy(X,band_width=100):
    num_bins = np.round(np.nanmax(X)-np.nanmin(X)/band_width)
    hist, bin_edges = np.histogram(X.flatten(), \
                                   bins=num_bins.astype(int), \
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
    
    Notes
    -----
    [1] Zhou et al. "Shadow detection and compensation from remote sensing 
    images under complex urban conditions", 2021.
    """  
    
    Y, Cb, Cr = rgb2ycbcr(Red, Green, Blue)    
    SI = np.divide(Cb-Y, Cb+Y)
#    Y, Cb, Cr = rgb2ycbcr(Red, Green, Blue)    
#    SI = np.divide(Cr+1, Y+1)    
#    H, C, V = rgb2hcv(Red, Green, Blue)    
#    SI = np.divide(H+1, V+1)
#    Y, I, Q = rgb2yiq(Red, Green, Blue)    
#    SI = np.divide(Q+1, Y+1)
#    H, S, I = rgb2hsi(Red, Green, Blue)    
#    SI = np.divide(H+1, I+1)  
    return SI

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
    
    Notes
    -----
    [1] Zhou et al. "Shadow detection and compensation from remote sensing 
    images under complex urban conditions", 2021.
    """        
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
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{SEI}=\frac{(B_{1}+B_{9})-(B_{3}+B_{8})}{(B_{1}+B_{9})+(B_{3}+B_{8})} 
    
    [1] Sun et al. "Combinational shadow index for building shadow extraction 
    in urban areas from Sentinel-2A MSI imagery" International journal of 
    applied earth observation and geoinformation, vol. 78 pp. 53--65, 2019.
    """   
    SEI = np.divide( (Blue + Near)-(Green + Red), (Blue + Near)+(Green + Red))
    return SEI

def false_color_shadow_difference_index(Blue,Green,Red,Near):
    """transform red and near infrared arrays to shadow index
    
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
    FCSI : np.array, size=(m,n)
        array with shadow enhancement
        
    See also
    --------
    normalized_sat_value_difference_index   
    
    Notes
    -----
    [1] Teke et al. "Multi-spectral false color shadow detection" Proceedings
    of the ISPRS conference on photogrammetric image analysis, pp.109-119, 
    2011.
    """ 
    
    (H, S, I) = rgb2hsi(Near, Red, Green)  # create HSI bands
    NSVDI = (S - I) / (S + I)
    return NSVDI
    
    SEI = np.divide( (Blue + Near)-(Green + Red), (Blue + Near)+(Green + Red))
    return SEI

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
    
    .. math:: \text{NDWI}=\frac{B_{3}-B_{8}}{B_{3}+B_{8}} 
    
    [1] Gao, "NDWI - a normalized difference water index for remote sensing of 
    vegetation liquid water from space" Remote sensing of environonment, 
    vol. 58 pp. 257–266, 1996
    """   
    NDWI = np.divide( (Green - Near), (Green + Near))
    return NDWI

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
    
    Notes
    -----
    [1] Sun et al. "Combinational shadow index for building shadow extraction 
    in urban areas from Sentinel-2A MSI imagery" International journal of 
    applied earth observation and geoinformation, vol. 78 pp. 53--65, 2019.
    """
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
    ISI : np.array, size=(m,n)
        array with shadow transform

    See also
    --------
    false_color_shadow_difference_index
    
    Notes
    -----
    [1] Ma et al. "Shadow segmentation and compensation in high resolution 
    satellite images", Proceedings of IEEE IGARSS, pp. II-1036--II-1039, 2008.
    """    
    (H, S, I) = rgb2hsi(Red, Green, Blue)  # create HSI bands
    NSVDI = (S - I) / (S + I)
    return NSVDI

def shadow_index_liu(Red, Green, Blue):
    """transform red, green, blue arrays to shadow index
    
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
        array with shadow enhancement
    
    Notes
    -----     
    [1] Liu & Xie. "Study on shadow detection in high resolution remote sensing
    image of PCA and HIS model", Remote sensing technology and application, 
    vol.28(1) pp. 78--84, 2013.    
    [1] Sun et al. "Combinational shadow index for building shadow extraction 
    in urban areas from Sentinel-2A MSI imagery" International journal of 
    applied earth observation and geoinformation, vol. 78 pp. 53--65, 2019.
    """     
    (H,S,I) = rgb2hsi(Red, Green, Blue)    

    X = pca_rgb_preparation(Red, Green, Blue, min_samp=1e4)
    e, lamb = pca(X)
    del X
    PC1 = np.dot(e[0, :], np.array([Red, Green, Blue]))
    
    SI = np.divide((PC1 - I) * (1 + S), (PC1 + I + S))
    return SI

def specthem_ratio(Red, Green, Blue):
    """transform red, green, blue arrays to specthem ratio
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image    
    Green : np.array, size=(m,n)
        green band of satellite image
    Blue : np.array, size=(m,n)
        blue band of satellite image
        
    Returns
    -------
    Sr : np.array, size=(m,n)
        array with shadow enhancement
    
    Notes
    -----
    In the paper the logarithmic is also used further enhance the shadows,
    
    .. math:: \tilde{\text{Sr}}=\log{\text{Sr}+1} 
    
    [1] Silva et al. "Near real-time shadow detection and removal in aerial 
    motion imagery application" ISPRS journal of photogrammetry and remote
    sensing, vol.140 pp.104--121, 2018.
    """    
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
    
    Notes
    -----
    [1] Mustafa & Abedehafez. "Accurate shadow detection from high-resolution 
    satellite images" IEEE geoscience and remote sensing letters, vol.14(4) 
    pp. 494--498, 2017.
    """    
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
    
    Notes
    -----
    [1] Mustafa & Abedelwahab. "Corresponding regions for shadow restoration 
    in satellite high-resolution images" International journal of remote 
    sensing, vol.39(20) pp.7014--7028, 2018.
    """                 
    (H, S, I) = erdas2hsi(Blue, RedEdge, Near)
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
    
    Notes
    -----
    [1] Han et al. "A mixed property-based automatic shadow detection approach 
    for VHR multispectral remote sensing images" Applied sciences vol.8(10) 
    pp.1883, 2018.
    """    
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
    
    Notes
    -----
    [1] Gevers & Smeulders. "Color-based object recognition" Pattern 
    recognition, vol.32 pp.453--464, 1999 
    [2] Arévalo González & G. Ambrosio. "Shadow detection in colour 
    high‐resolution satellite images" International journal of remote sensing, 
    vol.29(7) pp.1945--1963, 2008
    """ 
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
        
    Returns
    -------
    c3 : np.array, size=(m,n)
        array with shadow enhancement
    
    Notes
    -----
    [1] Gevers & Smeulders. "Color-based object recognition" Pattern 
    recognition, vol.32 pp.453--464, 1999 
    [2] Besheer & G. Abdelhafiz. "Modified invariant colour model for shadow 
    detection" International journal of remote sensing, vol.36(24) 
    pp.6214--6223, 2015.
    10.1080/01431161.2015.1112930
    """     
    C3 = color_invariant(Blue, Green, Red)
    NDWI = normalized_difference_water_index(Green, Near)
    
    # [2] Uses thresholding an boolean algebra, 
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
       
    Notes
    -----
    [1] Reinhard et al. "Color transfer between images" IEEE Computer graphics 
    and applications vol.21(5) pp.34-41, 2001.
    """
    (L,M,S) = rgb2lms(Red, Green, Blue)
    L = np.log10(L)
    M = np.log10(M)
    S = np.log10(S)
    (l,alfa,beta) = lms2lab(L, M, S)
    
    return l

def finlayson(Ia, Ib, Ic, a=-1):
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
       
    Notes
    -----
    [1] Finlayson et al. "Entropy minimization for shadow removal" 
    International journal of computer vision vol.85(1) pp.35-57 2009.
    """
    Ia[Ia==0] = 1e-6    
    Ib[Ib==0] = 1e-6    
    sqIc = np.power(Ic, 1./2)
    sqIc[sqIc==0] = 1    
        
    chi1 = np.log(np.divide( Ia, sqIc))
    chi2 = np.log(np.divide( Ib, sqIc))
    
    if a==-1: # estimate angle from data if angle is not given
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
        #a = angl[np.argmax(shan)]
    # create imagery    
    S = - np.cos(np.radians(a))*chi1 - np.sin(np.radians(a))*chi2
    #a += 90
    #R = np.cos(np.radians(a))*chi1 + np.sin(np.radians(a))*chi2
    return S

def shade_index(Blue, Green, Red, Near):
    """transform blue, green, red and near infrared arrays to shade index
    
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
    l : np.array, size=(m,n)
        shadow enhanced band
       
    Notes
    -----
    Amplify dark regions by simple multiplication:
        
    .. math:: \text{SI}= \Pi_{i}^{N}{1 - B_{i}}    
        
    [1] Altena "Filling the white gap on the map: Photoclinometry for glacier 
    elevation modelling" MSc thesis TU Delft, 2012.
    """
    SI = np.prod(np.dstack([(1 - Red), (1 - Green), (1 - Blue), (1 - Near)]),
                 axis=2)
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
       
    Notes
    -----   
    [1] Fredembach & Süsstrunk. "Automatic and accurate shadow detection from 
    (potentially) a single image using near-infrared information" 
    EPFL Tech Report 165527, 2010.
    [2] Rüfenacht et al. "Automatic and accurate shadow detection using 
    near-infrared information" IEEE transactions on pattern analysis and 
    machine intelligence, vol.36(8) pp.1672-1678, 2013.
    """
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
# histogram matching....

# TO DO:
# Wu 2007, Natural Shadow Matting. DOI: 10.1145/1243980.1243982
# https://github.com/yaksoy/AffinityBasedMattingToolbox
