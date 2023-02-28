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