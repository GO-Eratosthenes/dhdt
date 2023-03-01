import numpy as np

def floating_algae_index(Red, Near, Swir, df):
    """ Following [Hu09]_

    Parameters
    ----------
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image
    Swir : numpy.ndarray, size=(m,n)
        shortwave infrared band of satellite image
    df : datafram

    Returns
    -------
    FAI : numpy.ndarray, size=(m,n)
        array with floating algae index

    References
    ----------
    .. [Hu09] Hu, "A novel ocean color index to detect floating algae in the
              global oceans" Remote sensing of the enviroment, vol.113(10)
              pp.2188-2129, 2009
    """
    IN = df['common_name'].isin(['red'])
    lambda_red = df[IN].center_wavelength.to_numpy()[0]
    IN = df['common_name'].isin(['nir'])
    lambda_nir = df[IN].center_wavelength.to_numpy()[0]
    IN = df['common_name'].isin(['swir16'])
    lambda_swi = df[IN].center_wavelength.to_numpy()[0]

    Red_prime = Red + np.multiply((Swir - Red),
                                  np.divide(lambda_nir-lambda_red,
                                            lambda_swi-lambda_red))
    FAI = Near - Red_prime
    return FAI

def alternative_floating_algae_index(Red, Rededge, Near):
    """transform near and short wave infrared arrays to get a algae-index.
    Based upon [Wa18]_

    Parameters
    ----------
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Rededge : numpy.ndarray, size=(m,n)
        rededge band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    AFAI : numpy.ndarray, size=(m,n)
        array with floating algae transform

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: AFAI=(B_{06} - B_{04})/(B_{8A} - B_{04})


    References
    ----------
    .. [Wa18] Wang, et al. "Remote sensing of Sargassum biomass, nutrients, and
              pigments" Geophysical Research Letters, vol.45(22) pp.12-359, 2018.
    """
    AFAI = np.divide( Near - Red, Rededge - Red)
    return AFAI

def normalized_difference_vegetation_index(Red, Near):
    """

    Parameters
    ----------
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    NDVI : numpy.ndarray, size=(m,n)
        array with normalized difference vegetation index

    References
    ----------
    .. [Ro74] Rouse et al. "Monitoring the vernal advancements and
              retrogradation of natural vegetation" in NASA/GSFC final report,
              pp.1–137, 1974.
    """
    NDVI = np.divide( Near - Red, Near + Red)
    return NDVI

def infrared_percentage_vegetation_index(Red, Near):
    """ adjustment over the NDVI, see [Cr90]_.

    Parameters
    ----------
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    NPVI : numpy.ndarray, size=(m,n), range=0...1
        array with infrared percentage vegetation index

    References
    ----------
    .. [Cr90] Crippen "Calculating the vegetation index faster" Remote sensing
              of environment, vol.34 pp.71-73, 1990.
    """
    NPVI = np.divide( Near, Near + Red)
    return NPVI

def renormalized_difference_vegetation_index(Red, Near):
    """ Based upon [RB95]_

    Parameters
    ----------
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    NDVI : numpy.ndarray, size=(m,n)
        array with renormalized difference vegetation index

    References
    ----------
    .. [RB95] Rougean & Breon, "Estimating PAR absorbed by vegetation from
              bidirectional reflectance measurements" Remote sensing of
              environment, vol.51 pp.375–384, 1995.
    """
    NDVI = np.divide( Near - Red, np.sqrt(Near + Red))
    return NDVI

def simple_ratio(Red, Near):
    """ Based upon [Jo69]_

    Parameters
    ----------
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    SR : numpy.ndarray, size=(m,n)
        array with simple ratio

    References
    ----------
    .. [Jo69] Jordan, "Derivation of leaf area index from quality of light on
              the forest floor" Ecology vol.50 pp.663–666, 1969.
    """
    SR = np.divide( Red, Near)
    return SR

def modified_simple_ratio(Red, Near):
    """ Based upon [Ch96]_

    Parameters
    ----------
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    SR : numpy.ndarray, size=(m,n)
        array with simple ratio

    References
    ----------
    .. [Ch96] Chen, "Evaluation of vegetation indices and modified simple ratio
              for boreal applications" Canadian journal of remote sensing,
              vol.22 pp.229–242, 1996.
    """
    MSR = np.divide( np.divide( Red, Near) -1,
                     np.sqrt( np.divide( Red, Near) +1 )
                   )
    return MSR

def visible_atmospherically_resistant_index(Blue, Green, Red):
# Novel algorithms for remote estimation of vegetation fraction
    VARI = np.divide(Green - Red, Green + Red - Blue)
    return VARI

def soil_adjusted_vegetation_index(Red, Near, L=.5):
    """ Based upon [Hu88]_

    Parameters
    ----------
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image
    L : float, default=.5
        multiplication factor

    Returns
    -------
    SAVI : numpy.ndarray, size=(m,n)
        array with soil adjusted vegetation index

    References
    ----------
    .. [Hu88] Huete, "A soil vegetation adjusted index (SAVI)" Remote sensing of
              environment, vol.25 pp.295–309, 1988.
    """
    SAVI = np.divide(np.multiply(1+L, Near - Red),
                     Near + Red + L)
    return SAVI

def modified_soil_adjusted_vegetation_index(Red, Near):
    """ Based upon [Qi94]_

    Parameters
    ----------
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    MSAVI : numpy.ndarray, size=(m,n)
        array with modified soil adjusted vegetation index

    References
    ----------
    .. [Qi94] Qi et al., "A modified soil vegetation adjusted index" Remote
              sensing of environment, vol.48, pp.119–126, 1994.
    """
    MSAVI = .5* (2*Near +1 -
                 np.sqrt( (2*Near + 1)**2 -
                         8*(Near - Red))
                 )
    return MSAVI

def atmospherically_resistant_vegetation_index(Blue, Red, Near, gamma=1.,L=.5):
    """ Based upon [KT92]_

    Parameters
    ----------
    Blue : numpy.ndarray, size=(m,n)
        blue band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Near : numpy.ndarray, size=(m,n)
        near infrared band of satellite image

    Returns
    -------
    ARVI : numpy.ndarray, size=(m,n)
        array with atmospherically resistant vegetation index

    References
    ----------
    .. [KT92] Kaufman & Tanre, "Atmospherically resistant vegetation index
              (ARVI)" IEEE transactions in geoscience and remote sensing,
              vol.30 pp.261–270, 1992.
    """
    RB = Red - gamma*(Blue - Red)
    ARVI = np.divide(np.multiply(1+L, Near - RB),
                     Near + RB + L)
    return ARVI

def modified_chlorophyll_absorption_index(Green, Red, Rededge):
    """ Based upon [Da00]_

    Parameters
    ----------
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Rededgde : numpy.ndarray, size=(m,n)
        rededge band of satellite image

    Returns
    -------
    MCARI : numpy.ndarray, size=(m,n)
        array with atmospherically resistant vegetation index

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: MCARI= ((B_{05} - B_{04}) - .2(B_{05} - B_{03})) *
                (B_{04})/(B_{03})

    References
    ----------
    .. [Da00] Daughtry et al. "Estimating corn leaf chlorophyll concentration
              from leaf and canopy reflectance" Remote sensing of environment
              vol.74 pp.229–239, 2000.
    """
    # Rededge B05 700, Red B04 670, Green B03 550
    MCARI = ((Rededge - Red) - .2*(Rededge - Green))*np.divide(Red, Green)
    return MCARI

def triangular_vegetation_index(Green, Red, Rededge):
    """ Based upon [BL00]_

    Parameters
    ----------
    Green : numpy.ndarray, size=(m,n)
        green band of satellite image
    Red : numpy.ndarray, size=(m,n)
        red band of satellite image
    Rededgde : numpy.ndarray, size=(m,n)
        rededge band of satellite image

    Returns
    -------
    TVI : numpy.ndarray, size=(m,n)
        array with triangular vegetation index

    Notes
    -----
    Based on the bands of Sentinel-2:

    .. math:: TVI= .5 (120(B_{06} - B_{03}) - 200(B_{04} - B_{03}))

    References
    ----------
    .. [BL00] Broge & Leblanc, "Comparing prediction power and stability of
              broadband and hyperspectral vegetation indices for estimation
              of green leaf area index and canopy chlorophyll density" Remote
              sensing of environment, vol.76 pp.156–172, 2000.
    """
    # Rededge B06 750, Red B04 670, Green B03 550
    TVI = .5* (120*(Rededge - Green) - 200*(Red - Green))
    return TVI

# see https://www.usgs.gov/core-science-systems/nli/landsat/landsat-enhanced-vegetation-index?qt-science_support_page_related_con=0#qt-science_support_page_related_con
def enhanced_vegetation_index(Blue,Red,Near,
                              G=2.5, L=1, C1=6., C2=7.5):
    EVI = G * np.divide(Near - Red , Near + C1*Red - C2*Blue + L)
    return EVI