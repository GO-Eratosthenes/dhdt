from dhdt.input.read_rapideye import read_band_re
from dhdt.input.read_sentinel2 import read_band_s2


def get_shadow_bands(satellite_name):
    """ give bandnumbers of visible bands of specific satellite

    If the instrument has a high resolution panchromatic band,
    this is given as the fifth entry

    Parameters
    ----------
    satellite_name : {'Sentinel-2', 'Landsat8', 'Landsat7', 'Landsat5',
                      'RapidEye', 'PlanetScope', 'ASTER', 'Worldview3'}
        name of the satellite or instrument abreviation.

    Returns
    -------
    band_num : list, size=(1,4), integer

        * band_num[0] : Blue band number
        * band_num[1] : Green band number
        * band_num[2] : Red band number
        * band_num[3] : Near-infrared band number
        * band_num[4] : panchrometic band

    See Also
    --------
    read_shadow_bands
    """
    satellite_name = satellite_name.lower()

    # compare if certain segments are present in a string

    if len([n for n in ['sentinel', '2'] if n in satellite_name]) == 2 or \
            len([n for n in ['msi'] if n in satellite_name]) == 1:
        band_num = [2, 3, 4, 8]
    elif len([n for n in ['landsat', '8'] if n in satellite_name]) == 2 or \
            len([n for n in ['oli'] if n in satellite_name]) == 1:
        band_num = [2, 3, 4, 5, 8]
    elif len([n for n in ['landsat', '7'] if n in satellite_name]) == 2 or \
            len([n for n in ['etm+'] if n in satellite_name]) == 1:
        band_num = [1, 2, 3, 4, 8]
    elif len([n for n in ['landsat', '5'] if n in satellite_name]) == 2 or \
            len([n for n in ['tm', 'mss'] if n in satellite_name]) == 1:
        band_num = [1, 2, 3, 4]
    elif len([n for n in ['rapid', 'eye'] if n in satellite_name]) == 2:
        band_num = [1, 2, 3, 5]
    elif len([n for n in ['planet'] if n in satellite_name]) == 1 or \
            len([n for n in ['dove'] if n in satellite_name]) == 1:
        band_num = [1, 2, 3, 5]
    elif len([n for n in ['aster'] if n in satellite_name]) == 1:
        band_num = [1, 1, 2, 3]
    elif len([n for n in ['worldview', '3'] if n in satellite_name]) == 2:
        band_num = [2, 3, 5, 8]

    return band_num


def read_shadow_bands(sat_path, band_num):
    """ read the specific band numbers of the multispectral satellite images

    Parameters
    ----------
    sat_path : string
        path to imagery and file name.
    band_num : list, size=(1,4), integer

        * band_num[0] : Blue band number
        * band_num[1] : Green band number
        * band_num[2] : Red band number
        * band_num[3] : Near-infrared band number
        * band_num[4] : panchrometic band

    Returns
    -------
    Blue : numpy.ndarray, size=(m,n), dtype=integer
        blue band of satellite image
    Green : numpy.ndarray, size=(m,n), dtype=integer
        green band of satellite image
    Red : numpy.ndarray, size=(m,n), dtype=integer
        red band of satellite image
    Near : numpy.ndarray, size=(m,n), dtype=integer
        near-infrared band of satellite image
    crs : string
        osr.SpatialReference in well known text
    geoTransform : tuple, size=(6,1)
        affine transformation coefficients.
    targetprj : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)
    Pan :
        Panchromatic band

        * numpy.ndarray, size=(m,n), integer,
        * None if band does not exist

    See Also
    --------
    get_shadow_bands

    """
    if len([n for n in ['S2', 'MSIL1C'] if n in sat_path]) == 2:
        # read imagery of the different bands
        (Blue, crs, geoTransform,
         targetprj) = read_band_s2(format(band_num[0], '02d'), sat_path)
        (Green, crs, geoTransform,
         targetprj) = read_band_s2(format(band_num[1], '02d'), sat_path)
        (Red, crs, geoTransform,
         targetprj) = read_band_s2(format(band_num[2], '02d'), sat_path)
        (Near, crs, geoTransform,
         targetprj) = read_band_s2(format(band_num[3], '02d'), sat_path)
        Pan = None
    elif len([n for n in ['RapidEye', 'RE'] if n in sat_path]) == 2:
        # read single imagery and extract the different bands
        (Blue, crs, geoTransform,
         targetprj) = read_band_re(format(band_num[0], '02d'), sat_path)
        (Green, crs, geoTransform,
         targetprj) = read_band_re(format(band_num[1], '02d'), sat_path)
        (Red, crs, geoTransform,
         targetprj) = read_band_re(format(band_num[2], '02d'), sat_path)
        (Near, crs, geoTransform,
         targetprj) = read_band_re(format(band_num[3], '02d'), sat_path)
        Pan = None

    return Blue, Green, Red, Near, crs, geoTransform, targetprj, Pan
