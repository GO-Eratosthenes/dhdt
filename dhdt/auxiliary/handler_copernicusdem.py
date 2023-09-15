# functions of use for the CopernicusDEM data

# generic libraries
import os

# import geospatial libraries
import pyproj
import rioxarray
import shapely.geometry
import shapely.ops
from affine import Affine
from rasterio.enums import Resampling
from rioxarray.merge import merge_arrays

# import local functions
from dhdt.auxiliary.handler_mgrs import get_geom_for_tile_code
from dhdt.generic.handler_sentinel2 import get_generic_s2_raster
from dhdt.generic.handler_www import get_file_from_www
from dhdt.generic.mapping_tools import get_bbox

DEM_DIR_DEFAULT = os.path.join('.', 'data', 'DEM')


def get_itersecting_DEM_tile_URLs(geometry, resolution=30):
    """
    Find the DEM tiles that intersect the geometry and returns the
    corresponding URLs. NOTE: the CRS of the geometry needs to be WGS84!

    Parameters
    ----------
    geometry : shapely.geometry.Polygon
        polygon of interest.
    resolution : int, {30, 90}
        input DEM resolution in meters, default 30

    Returns
    -------
    list of strings
        file name of tiles of interest.

    Notes
    -----
    The following acronyms are used:

    - CopDEM : Copernicus DEM
    - CRS : coordinate reference system
    - DEM : digital elevation model
    """
    minx, miny, maxx, maxy = geometry.bounds
    lat_grid_min = int(miny // 1)
    lat_grid_max = int(maxy // 1)
    lon_grid_min = int(minx // 1)
    lon_grid_max = int(maxx // 1)
    url_list = [
        get_DEM_tile_URL(lat_grid, lon_grid, resolution=resolution)
        for lat_grid in range(lat_grid_min, lat_grid_max + 1)
        for lon_grid in range(lon_grid_min, lon_grid_max + 1)
    ]
    return url_list


def get_DEM_tile_URL(lat, lon, resolution=30):
    """
    Construct the DEM tile URL from its latitude and longitude grid values.
    Data is organized in tiles in a 1x1 degree grid. For more information see
    [copDEM-aws]_.

    Parameters
    ----------
    lat : int
        latitude grid value (-90 to 89)
    lon : int
        longitude grid value (-180 to 179)
    resolution : int, {30, 90}
        input DEM resolution in meters, default 30

    Returns
    -------
    string
        DEM tile URL

    References
    ----------
    .. [copDEM-aws] https://copernicus-dem-30m.s3.amazonaws.com/readme.html
    """
    assert resolution in {30, 90}, "resolution not available"
    lat_str = '{:s}{:02d}'.format('N' if lat > 0 else 'S', abs(lat))
    lon_str = '{:s}{:03d}'.format('E' if lon > 0 else 'W', abs(lon))
    dem_root_url = (
        f'https://copernicus-dem-{resolution}m.s3.eu-central-1.amazonaws.com')
    res_arc_secs = {30: 10, 90: 30}[resolution]  # use resolution in arc secs
    stem = f'Copernicus_DSM_COG_{res_arc_secs}_{lat_str}_00_{lon_str}_00_DEM'
    return '/'.join([dem_root_url, stem, f'{stem}.tif'])


def make_copDEM_mgrs_tile(tile_code,
                          resolution=10,
                          input_resolution=30,
                          dem_dir=None,
                          tile_path=None,
                          out_dir=None):
    """
    Download Copernicus DEM data and retile it according to the MGRS scheme,
    using the suitable UTM CRS.

    Parameters
    ----------
    tile_code : string
        MGRS code of the tile of interest
    resolution : int
        output DEM resolution in meters, default 10
    input_resolution : int, {30, 90}
        input DEM resolution in meters, default 30
    dem_dir : string
        location where the elevation data is situated (or will be downloaded)
    tile_path : string
        path to the MGRS tiling file
    out_dir : string
        location where the retiled DEM file can be written

    Returns
    -------
    string
        location and filename of the retiled DEM raster

    Notes
    -----
    The following acronyms are used:

    - CopDEM : Copernicus DEM
    - CRS : coordinate reference system
    - DEM : digital elevation model
    - MGRS : US Military Grid Reference System
    - UTM : Universal Transverse Mercator
    """

    dem_dir = DEM_DIR_DEFAULT if dem_dir is None else dem_dir
    out_dir = dem_dir if out_dir is None else out_dir

    geoTransform, crs = get_generic_s2_raster(tile_code,
                                              spac=resolution,
                                              tile_path=tile_path)

    cop_dem = get_copDEM_in_raster(geoTransform, crs, input_resolution,
                                   dem_dir)

    cop_dem_out = os.path.join(out_dir, tile_code + '.tif')
    cop_dem.rio.to_raster(cop_dem_out,
                          tiled=True,
                          compress='LZW',
                          dtype=cop_dem.dtype)
    return cop_dem_out


def get_copDEM_in_raster(geoTransform, crs, input_resolution=30, dem_dir=None):
    """
    Download and retile Copernicus DEM data.

    Parameters
    ----------
    geoTransform : tuple
        georeference transform of the output raster, including shape
    crs : pyproj.crs.crs.CRS
        coordinate reference system (CRS) of the output raster
    input_resolution : int, {30, 90}
        input DEM resolution in meters, default 30
    dem_dir : string
        location where the elevation data is situated (or will be downloaded)
    Returns
    -------
    xarray.core.dataarray.DataArray
        Retiled raster DEM
    """
    bbox = get_bbox(geoTransform).reshape((2, 2)).T.ravel()
    polygon = shapely.geometry.box(*bbox, ccw=True)

    # reproject to lat, lon
    crs_wgs84 = pyproj.CRS.from_epsg(4326)  # lat lon
    transformer = pyproj.Transformer.from_crs(crs_from=crs,
                                              crs_to=crs_wgs84,
                                              always_xy=True)
    polygon_wgs84 = shapely.ops.transform(transformer.transform, polygon)

    dem_tiles = get_itersecting_DEM_tile_URLs(polygon_wgs84,
                                              resolution=input_resolution)
    cop_dem = download_and_mosaic(dem_tiles,
                                  crs,
                                  geoTransform,
                                  dem_dir=dem_dir)
    # Z_cop = np.squeeze(cop_dem.values)
    # Msk = Z_cop>3.4E38
    # Z_cop = np.ma.array(Z_cop, mask=Msk)
    return cop_dem


def get_copDEM_s2_tile_intersect_list(tile_code, tile_path=None):
    """
    Get a list of URLs of DEM tiles that are touching or in a Sentinel-2 tile

    Parameters
    ----------
    tile_code : string
        MGRS Sentinel-2 tile code of interest
    tile_path : string
        path to the MGRS tiling file

    Returns
    -------
    urls : list of strings
        URLs of the relevant DEM tiles

    See Also
    --------
    .handler_sentinel2.get_tiles_from_s2_list

    Notes
    -----
    The following acronyms are used:

    - CopDEM : Copernicus DEM
    - MGRS : military gridded reference system
    - CDS : Copernicus data service
    - DEM : digital elevation model
    - DGED : defence gridded elevation data
    - DTED : digital terrain elevation data
    """
    geom = get_geom_for_tile_code(tile_code, tile_path)
    return get_itersecting_DEM_tile_URLs(geom)


def download_and_mosaic(dem_tile_urls,
                        crs,
                        geoTransform,
                        shape=None,
                        dem_dir=None,
                        overwrite=False):
    """
    Download Copernicus DEM tiles and create mosaic according to satellite
    imagery tiling scheme.

    Parameters
    ----------
    dem_tile_urls : list of strings
        list of DEM tile URLs
    crs : string
        coordinate reference system of the output raster DEM
    geoTransform : tuple, size=(6,1)
        affine transformation coefficients of the output raster DEM
    shape : tuple
        shape of the output raster DEM. If not specified, assume the last two
        elements of the geoTransform represent the raster shape
    dem_dir : string
        location where the DEM tiles will be saved
    overwrite : bool
        If true, and DEM tiles are already present, download them again.

    Returns
    -------
    dem : xarray.core.dataarray.DataArray
        mosaicked DEM
    """
    print('Great you use CopernicusDEM, if you do use this data for ' +
          'products, publications, repositories, etcetera. \n' +
          'Please make sure you cite via the following doi: ' +
          'https://doi.org/10.5270/ESA-c5d3d65')
    dem_dir = DEM_DIR_DEFAULT if dem_dir is None else dem_dir

    dem_tile_paths = []
    for dem_tile_url in dem_tile_urls:
        dem_tile = get_file_from_www(dem_tile_url,
                                     dump_dir=dem_dir,
                                     overwrite=overwrite)
        dem_tile_paths.append(os.path.join(dem_dir, dem_tile))

    dem = mosaic_tiles(dem_tile_paths, crs, geoTransform, shape)
    return dem


def mosaic_tiles(dem_tile_paths, crs, geoTransform, shape=None):
    """
    Given a selection of elevation model tiles, mosaic them together

    Parameters
    ----------
    dem_tile_paths : list with strings
        list of DEM tile paths
    crs : string
        coordinate reference system
    geoTransform : tuple, size={(6,1), (8,1)}
        affine transformation coefficients of the DEM tile
    shape : tuple
        shape of the output raster DEM. If not specified, assume the last two
        elements of the geoTransform represent the raster shape

    Returns
    -------
    merged : xarray.core.dataarray.DataArray
        mosaicked DEM
    """
    if shape is None:
        assert len(geoTransform) == 8, 'output raster DEM shape not specified'
        shape = geoTransform[-2:]

    # geoTransform needs to be converted to Affine model
    afn = Affine.from_gdal(*geoTransform[:6])

    merged = None
    for f in dem_tile_paths:
        dem_tile = rioxarray.open_rasterio(f, masked=True)
        # reproject to image CRS
        dem_tile_xy = dem_tile.rio.reproject(crs,
                                             transform=afn,
                                             resampling=Resampling.lanczos,
                                             shape=shape,
                                             dtype=dem_tile.dtype,
                                             nodata=dem_tile.rio.nodata)

        if merged is None:
            merged = dem_tile_xy
        else:
            merged = merge_arrays([merged, dem_tile_xy],
                                  nodata=dem_tile.rio.nodata)
    return merged
