from osgeo import osr


def create_local_crs():
    """ create spatial refence of local horizontal datum

    Returns
    -------
    crs : {osr.SpatialReference, string}
        the coordinate reference system via GDAL SpatialReference description

    See Also
    --------
    dhdt.testing.terrain_tools.create_artificial_terrain
    dhdt.generic.unit_check.is_crs_an_srs
    """
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(8377)

    print('OBS: taking local coordinate system')
    return crs


def create_artificial_geoTransform(Z, spac=10.):
    """ create an dummy tuple, based on the size of an array and a pixel
    spacing

    Parameters
    ----------
    Z : numpy.ndarray
        grid of interested, that needs a geotransform
    spac : float
        pixel spacing

    Returns
    -------
    geoTransform : tuple, size=(8,)
        affine transformation coefficients, and image size.
    """
    if isinstance(spac, int):
        spac = float(spac)
    m, n = Z.shape[:2]
    geoTransform = (0., +spac, 0., 0., 0., -spac, m, n)
    return geoTransform
