import numpy as np

def snow_cover_fraction(Near, Short):
    r"""transform near and short wave infrared arrays to get a snow-index, from
    [Pa02]_.
    
    Parameters
    ----------    
    Near : numpy.array, size=(m,n)
        near infrared band of satellite image
    Short : numpy.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    NDSII : numpy.array, size=(m,n)
        array with NDSII transform
    
    Notes
    -----
    Based on the bands of Sentinel-2:

        .. math:: SCF= B_{8} / B_{11}

    The bands of Sentinel-2 correspond to the following spectral range:

        - :math:`B_{8}` : near infrared
        - :math:`B_{11}` : shortwave infrared

    References
    ----------
    .. [Pa02] Paul et al. "The new remote-sensing-derived Swiss glacier
              inventory: I methods" Annuals of glaciology, vol.34 pp.355-361,
              2002.
    """   
    SCF = np.divide( Near, Short)
    return SCF

def snow_fraction(Red, Short):
    r"""transform red and short wave infrared arrays to get a snow-index, see
    also [PK05]_.
    
    Parameters
    ----------    
    Red : numpy.array, size=(m,n)
        red band of satellite image
    Short : numpy.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    SF : numpy.array, size=(m,n)
        array with snow fraction transform
    
    Notes
    -----
    Based on the bands of Sentinel-2:

        .. math:: SF = B_{4} / B_{11}

    The bands of Sentinel-2 correspond to the following spectral range:

        - :math:`B_{4}` : red
        - :math:`B_{11}` : shortwave infrared

    References
    ----------
    .. [PK05] Paul & Kääb "Perspectives on the production of a glacier inventory
              from multispectral satellite data in Arctic Canada: Cumberland
              peninsula, Baffin Island" Annuals of glaciology, vol.42 pp.59-66,
              2005.
    """   
    SF = np.divide( Red, Short)
    return SF


def normalized_difference_snow_index(Green, Short):
    r"""transform red and short wave infrared arrays NDSII-index, see also
    [Do89]_.
    
    Parameters
    ----------    
    Green : numpy.array, size=(m,n)
        green band of satellite image
    Short : numpy.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    NDSI : numpy.array, size=(m,n)
        array with NDSI transform

    Notes
    -----
    Based on the bands of Sentinel-2:
        .. math:: NDSI = (B_{3}-B_{8}) / (B_{3}+B_{8})

    The bands of Sentinel-2 correspond to the following spectral range:
        - :math:`B_{3}` : green
        - :math:`B_{8}` : near infrared

    See also
    --------
    normalized_difference_snow_ice_index

    References
    ----------
    .. [Do89] Dozier. "Spectral signature of alpine snow cover from the Landsat
              Thematic Mapper" Remote sensing of environment, vol.28 pp.9-22,
              1989.
    """   
    NDSI = np.divide( (Green - Short), (Green + Short))
    return NDSI

def normalized_difference_snow_ice_index(Red, Short):
    r"""transform red and short wave infrared arrays NDSII-index, see also [Xi01]
    
    Parameters
    ----------    
    Red : numpy.array, size=(m,n)
        red band of satellite image
    Short : numpy.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    NDSII : numpy.array, size=(m,n)
        array with NDSII transform

    Notes
    -----
    Based on the bands of Sentinel-2:
    
        .. math:: NDSII = (B_{4}-B_{11}) / (B_{4}+B_{11})

    The bands of Sentinel-2 correspond to the following spectral range:

        - :math:`B_{4}` : red
        - :math:`B_{11}` : shortwave infrared

    See also
    --------
    normalized_difference_snow_index, snow_fraction

    References
    ----------
    .. [Xi01] Xiao et al. "Assessing the potential of Vegetation sensor data for
              mapping snow and ice cover: A normalized snow and ice index"
              International journal of remote sensing, vol.22(13) pp.2479-2487,
              2001.
    """   
    NDSII = np.divide( (Red - Short), (Red + Short))
    return NDSII

def s3(Red, Near, Short):
    r"""transform red, near and shortwave infrared arrays to s3, see [SY99]_.
    
    Parameters
    ----------    
    Red : numpy.array, size=(m,n)
        red band of satellite image
    Near : numpy.array, size=(m,n)
        near infrared band of satellite image
    Short : numpy.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    S3 : numpy.array, size=(m,n)
        array with S3 transform

    See also
    --------
    snow_water_index
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
        .. math:: S3 = B_{8}(B_{4}-B_{11}) / ((B_{4} + B_{8})(B_{4} + B_{11}))

    The bands of Sentinel-2 correspond to the following spectral range:

        - :math:`B_{4}` : red
        - :math:`B_{8}` : near infrared
        - :math:`B_{11}` : shortwave infrared

    See Also
    --------
    snow_water_index

    References
    ----------
    .. [SY99] Saito & Yamazaki. "Characteristics of spectral reflectance for
              vegetation ground surfaces with snowcover; Vegetation indeces and
              snow indices" Journal of Japan society of hydrology and water
              resources, vol.12(1) pp.28-38, 1999.
    """   
    S3 = np.divide( Near*(Red - Short), (Red + Near)*(Red + Short))
    return S3

def snow_water_index(Green, Near, Short):
    r"""transform green, near and shortwave infrared arrays to snow water index,
    as described in [Di19]_.
    
    Parameters
    ----------    
    Green : numpy.array, size=(m,n)
        green band of satellite image
    Near : numpy.array, size=(m,n)
        near infrared band of satellite image
    Short : numpy.array, size=(m,n)
        shortwave infrared band of satellite image
        
    Returns
    -------
    SWI : numpy.array, size=(m,n)
        array with SWI transform
    
    See also
    --------
    s3
    
    Notes
    -----
    Based on the bands of Sentinel-2:

        .. math:: SWI = B_{8}(B_{3}-B_{11}) / ((B_{3} + B_{8})(B_{3} + B_{11}))

    The bands of Sentinel-2 correspond to the following spectral range:

        - :math:`B_{3}` : green
        - :math:`B_{8}` : near infrared
        - :math:`B_{11}` : shortwave infrared

    References
    ----------
    .. [Di19] Dixit et al. "Development and evaluation of a new snow water index
              (SWI) for accurate snow cover delineation" Remote sensing,
              vol.11(23) pp.2774, 2019.
    """   
    SWI = np.divide( Green*(Near - Short), (Green + Near)*(Near + Short))
    return SWI

def automated_glacier_extraction_index(Red, Near, Short, alpha=0.5):
    """transform red, near and shortwave infrared arrays to AGEI, based on
    [Zh19]_.
    
    Parameters
    ----------    
    Red : numpy.array, size=(m,n)
        red band of satellite image
    Near : numpy.array, size=(m,n)
        near infrared band of satellite image
    Short : numpy.array, size=(m,n)
        shortwave infrared band of satellite image
    alpha : float
        
    Returns
    -------
    AGEI : numpy.array, size=(m,n)
        array with AGEI transform

    See also
    --------
    snow_cover_fraction, snow_fraction
    
    Notes
    -----
    Based on the bands of Sentinel-2:
    
    .. math:: AGEI = (B_{3}-B_{8}) / (B_{3}+B_{8})
    
    References
    ----------
    .. [Zh19] Zhang et al. "Automated glacier extraction index by optimization
              of red/SWIR and NIR/SWIR ratio index for glacier mapping using
              Landsat imagery" Water, vol.11 pp.1233, 2019.
    """   
    AGEI = np.divide( alpha*Red + (1-alpha)*Near, Short)
    return AGEI
