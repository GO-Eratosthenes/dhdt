import numpy as np

def snow_cover_fraction(Near, Short):
    """transform near and short wave infrared arrays to get a snow-index
    
    Parameters
    ----------    
    Near : np.array, size=(m,n)
        near infrared band of satellite image
    Short : np.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    NDSII : np.array, size=(m,n)
        array with NDSII transform
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{SCF}=\frac{B_{8}}{B_{11}} 
    
    References
    ----------
    .. [1] Paul et al. "The new remote-sensing-derived Swiss glacier inventory: 
        I methods" Annuals of glaciology, vol.34 pp.355-361, 2002.
    """   
    SCF = np.divide( Near, Short)
    return SCF

def snow_fraction(Red, Short):
    """transform red and short wave infrared arrays to get a snow-index
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image
    Short : np.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    SF : np.array, size=(m,n)
        array with snow fraction transform
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{SF}=\frac{B_{4}}{B_{11}} 
    
    References
    ----------
    .. [1] Paul & Kääb "Perspectives on the production of a glacier inventory 
       from multispectral satellite data in Arctic Canada: Cumberland 
       peninsula, Baffin Island" Annuals of glaciology, vol.42 pp.59-66, 2005.
    """   
    SF = np.divide( Red, Short)
    return SF


def normalized_difference_snow_index(Green, Short):
    """transform red and short wave infrared arrays NDSII-index
    
    Parameters
    ----------    
    Green : np.array, size=(m,n)
        green band of satellite image
    Short : np.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    NDSI : np.array, size=(m,n)
        array with NDSI transform

    See also
    --------
    normalized_difference_snow_ice_index
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{NDSII}=\frac{B_{3}-B_{8}}{B_{3}+B_{8}} 
    
    References
    ----------
    .. [1] Dozier. "Spectral signature of alpine snow cover from the Landsat
       Thematic Mapper" Remote sensing of environment, vol.28 pp.9-22, 1989.
    """   
    NDSI = np.divide( (Green - Short), (Green + Short))
    return NDSI

def normalized_difference_snow_ice_index(Red, Short):
    """transform red and short wave infrared arrays NDSII-index
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image
    Short : np.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    NDSII : np.array, size=(m,n)
        array with NDSII transform

    See also
    --------
    normalized_difference_snow_index
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{NDSII}=\frac{B_{3}-B_{8}}{B_{3}+B_{8}} 
    
    References
    ----------
    .. [1] Xiao et al. "Assessing the potential of Vegetation sensor data for 
       mapping snow and ice cover: A normalized snow and ice index" 
       International journal of remote sensing, vol.22(13) pp.2479-2487, 2001.
    """   
    NDSII = np.divide( (Red - Short), (Red + Short))
    return NDSII

def s3(Red, Near, Short):
    """transform red, near and shortwave infrared arrays to s3
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image
    Short : np.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    S3 : np.array, size=(m,n)
        array with S3 transform

    See also
    --------
    snow_water_index
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{NDSII}=\frac{B_{3}-B_{8}}{B_{3}+B_{8}} 
    
    References
    ----------
    .. [1] Saito & Yamazaki. "Characteristics of spectral reflectance for 
       vegetation ground surfaces with snowcover; Vegetation indeces and snow
       indices" Journal of Japan society of hydrology and water resources, 
       vol.12(1) pp.28-38, 1999.
    """   
    S3 = np.divide( Near*(Red - Short), (Red + Near)*(Red + Short))
    return S3

def snow_water_index(Green, Near, Short):
    """transform red, near and shortwave infrared arrays to s3
    
    Parameters
    ----------    
    Green : np.array, size=(m,n)
        green band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image
    Short : np.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    SWI : np.array, size=(m,n)
        array with SWI transform
    
    See also
    --------
    s3
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{NDSII}=\frac{B_{3}-B_{8}}{B_{3}+B_{8}} 
    
    References
    ----------
    .. [1] Dixit et al. "Development and evaluation of a new snow water index 
       (SWI) for accurate snow cover delineation" Remote sensing, vol.11(23) 
       pp.2774, 2019.
    """   
    SWI = np.divide( Green*(Near - Short), (Green + Near)*(Near + Short))
    return SWI

def automated_glacier_extraction_index(Red, Near, Short, alpha=0.5):
    """transform red, near and shortwave infrared arrays to AGEI
    
    Parameters
    ----------    
    Red : np.array, size=(m,n)
        red band of satellite image
    Near : np.array, size=(m,n)
        near infrared band of satellite image
    Short : np.array, size=(m,n)
        shortwave infrared band of satellite image
    alpha : float
        
    Returns
    -------
    AGEI : np.array, size=(m,n)
        array with AGEI transform

    See also
    --------
    snow_cover_fraction, snow_fraction
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: \text{AGEI}=\frac{B_{3}-B_{8}}{B_{3}+B_{8}} 
    
    References
    ----------
    .. [1] Zhang et al. "Automated glacier extraction index by optimization of 
       red/SWIR and NIR/SWIR ratio index for glacier mapping using Landsat 
       imagery" Water, vol.11 pp.1233, 2019.
    """   
    AGEI = np.divide( alpha*Red + (1-alpha)*Near, Short)
    return AGEI
