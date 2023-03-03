# functions to work with the Randolph Glacier Inventory
import os

import affine
import geopandas as gpd
import pandas as pd
import rasterio.features
import rioxarray
import shapely.geometry
import shapely.wkt
import xarray as xr

from .handler_mgrs import get_tile_codes_from_geom, get_geom_for_tile_code
from ..generic.handler_www import get_file_from_www, get_file_and_extract
from ..generic.handler_sentinel2 import get_generic_s2_raster


# RGI version 6
RGI_V6_ROOT_URL = 'https://www.glims.org/RGI/rgi60_files'
RGI_V6_INDEX_URL = f'{RGI_V6_ROOT_URL}/00_rgi60_regions.zip'
RGI_V6_REGION_FILENAMES = [
    '01_rgi60_Alaska.zip',
    '02_rgi60_WesternCanadaUS.zip',
    '03_rgi60_ArcticCanadaNorth.zip',
    '04_rgi60_ArcticCanadaSouth.zip',
    '05_rgi60_GreenlandPeriphery.zip',
    '06_rgi60_Iceland.zip',
    '07_rgi60_Svalbard.zip',
    '08_rgi60_Scandinavia.zip',
    '09_rgi60_RussianArctic.zip',
    '10_rgi60_NorthAsia.zip',
    '11_rgi60_CentralEurope.zip',
    '12_rgi60_CaucasusMiddleEast.zip',
    '13_rgi60_CentralAsia.zip',
    '14_rgi60_SouthAsiaWest.zip',
    '15_rgi60_SouthAsiaEast.zip',
    '16_rgi60_LowLatitudes.zip',
    '17_rgi60_SouthernAndes.zip',
    '18_rgi60_NewZealand.zip',
    '19_rgi60_AntarcticSubantarctic.zip'
]


def _get_rgi_v6_region_urls(rgi_regions):
    rgi_codes = rgi_regions['RGI_CODE']
    filenames = [RGI_V6_REGION_FILENAMES[code-1] for code in rgi_codes]
    return [f'{RGI_V6_ROOT_URL}/{fname}' for fname in filenames]


def _get_rgi_v6_ids(rgi_glaciers):
    return rgi_glaciers['RGIId'].apply(lambda x: int(x.split('.')[-1]))


# RGI version 7
RGI_V7_ROOT_URL = (
    'https://cluster.klima.uni-bremen.de/~fmaussion/misc/rgi7_data'
)
RGI_V7_INDEX_URL = (
    f'{RGI_V7_ROOT_URL}/00_rgi70_regions/00_rgi70_O1Regions.zip'
)


def _get_rgi_v7_region_urls(rgi_regions):
    rgi_codes = rgi_regions['long_code']
    filenames = [f'RGI2000-v7.0-G-{code}.tar.gz' for code in rgi_codes]
    return [f'{RGI_V7_ROOT_URL}/l4_rgi7b0_tar/{fname}' for fname in filenames]


def _get_rgi_v7_ids(rgi_glaciers):
    return rgi_glaciers['rgi_id'].apply(lambda x: int(x.split('-')[-1]))


RGI_DATASETS = {
    6: {
        'index_url': RGI_V6_INDEX_URL,
        'get_rgi_region_urls': _get_rgi_v6_region_urls,
        'get_rgi_ids': _get_rgi_v6_ids,
    },
    7: {
        'index_url': RGI_V7_INDEX_URL,
        'get_rgi_region_urls': _get_rgi_v7_region_urls,
        'get_rgi_ids': _get_rgi_v7_ids,
    },
}


RGI_DIR_DEFAULT = os.path.join('.', 'data', 'RGI')


def _get_rgi_index_filename(version):
    index_url = RGI_DATASETS[version]['index_url']
    _, index_filename = os.path.split(index_url)
    return index_filename


def _get_rgi_shapes(rgi_paths, version):
    glaciers = pd.concat([gpd.read_file(rgi_path) for rgi_path in rgi_paths])
    get_rgi_ids = RGI_DATASETS[version]['get_rgi_ids']
    ids = get_rgi_ids(glaciers)
    glaciers = glaciers.set_index(ids)
    return glaciers.geometry


def which_rgi_regions(aoi=None, version=7, rgi_file=None):
    """
    Discover which Randolph Glacier Inventory (RGI) regions are used
    for a given region.

    Parameters
    ----------
    aoi : {iterable, string}, unit=degrees
        location of interest, given as (lat, lon) or as MGRS tile code
    version : {6,7}
        version of the glacier inventory
    rgi_file : string
        file path of the shapefile that describes the RGI regions

    Returns
    -------
    rgi_region_urls : list
        URLs of the relevant RGI regions

    Notes
    -----
    The following nomenclature is used:

    - lon : longitude, range=-180...+180, unit=degrees
    - lat : latitude, range=-90...+90, unit=degrees
    - RGI : Randolph glacier inventory
    - AOI : area of interest
    """
    assert isinstance(version, int), 'please provide an integer'
    assert version in RGI_DATASETS.keys(), f'version {version} not supported'
    if rgi_file is None:
        rgi_file = os.path.join(
            RGI_DIR_DEFAULT, _get_rgi_index_filename(version)
        )

    rgi_reg = gpd.read_file(rgi_file)

    if aoi is not None:
        if isinstance(aoi, str):
            geom_wkt = get_geom_for_tile_code(aoi)
            geom = shapely.wkt.loads(geom_wkt)
        else:
            geom = shapely.geometry.Point(aoi[1], aoi[0])  # lon, lat
        mask = rgi_reg.intersects(geom)
        rgi_reg = rgi_reg[mask]

    get_rgi_region_urls = RGI_DATASETS[version]['get_rgi_region_urls']
    if len(rgi_reg) == 0:
        raise IndexError('Selection not in Randolph glacier inventory')
    else:
        return get_rgi_region_urls(rgi_reg)


def download_rgi(aoi=None, version=7, rgi_dir=None, overwrite=False):
    """
    Download the relevant Randolph Glacier Inventory (RGI) files. If a region
    of interest is not provided, download all regions.

    Parameters
    ----------
    aoi : {iterable, string}, unit=degrees
        location of interest, given as (lat, lon) or as MGRS tile code
    version : {6,7}
        version of the glacier inventory
    rgi_dir : string
        location where the geospatial files of the RGI will be saved
    overwrite : bool
        if True and files exist, download them again, otherwise do nothing

    Returns
    -------
    rgi_paths : list
        paths of the downloaded RGI files

    Notes
    -----
    The following nomenclature is used:

    - lon : longitude, range=-180...+180, unit=degrees
    - lat : latitude, range=-90...+90, unit=degrees
    - RGI : Randolph glacier inventory
    - AOI : area of interest
    """
    rgi_dir = RGI_DIR_DEFAULT if rgi_dir is None else rgi_dir

    # download region index
    index_url = RGI_DATASETS[version]['index_url']
    index_file = get_file_from_www(index_url, rgi_dir, overwrite)

    # determine relevant region(s)
    rgi_region_urls = which_rgi_regions(
        aoi, version, os.path.join(rgi_dir, index_file)
    )

    # download region files
    shapefiles = []
    for url in rgi_region_urls:
        files = get_file_and_extract(url, rgi_dir, overwrite)
        paths = [f'{rgi_dir}/{f}' for f in files if f.endswith('.shp')]
        shapefiles.extend(paths)
    return shapefiles


def create_rgi_raster(rgi_shapes, geoTransform, crs, raster_path=None):
    """
    Creates a raster file in the location given by "raster_path", the content
    of which are the RGI glaciers in the input vector files

    Parameters
    ----------
    rgi_shapes : iterable
        RGI glacier shapes to rasterize
    geoTransform : tuple, size=(8,)
        georeference transform of an image.
    crs : osgeo.osr.SpatialReference() object
        coordinate reference system (CRS)
    raster_path : string
        location where the raster file should be positioned

    Notes
    -----
    The following nomenclature is used:

    RGI : Randolph glacier inventory
    CRS : Coordinate reference system
    """

    *transform_gdal, ny, nx = geoTransform
    transform = affine.Affine.from_gdal(*transform_gdal)
    crs_wkt = crs.ExportToWkt()

    rgi_shapes_repr = rgi_shapes.to_crs(crs_wkt)
    data = rasterio.features.rasterize(
        zip(rgi_shapes_repr.geometry, rgi_shapes_repr.index),
        out_shape=(ny, nx),
        fill=0,
        transform=transform,
        dtype=int,
    )

    raster = xr.DataArray(data=data, dims=('y', 'x'))
    raster.rio.write_nodata(0, inplace=True)
    raster.rio.write_crs(crs_wkt, inplace=True)
    raster.rio.write_transform(transform, inplace=True)

    os.makedirs(os.path.dirname(raster_path), exist_ok=True)
    raster.rio.to_raster(raster_path, tiled=True, compress='LZW')
    return


def create_rgi_tile_s2(aoi, version=7, rgi_dir=None, rgi_out_dir=None):
    """
    Creates a raster file with the extent of a generic Sentinel-2 tile, that is
    situated at a specific location

    Parameters
    ----------
    aoi : {iterable, string}, unit=degrees
        location of interest, given as (lat, lon) or as MGRS tile code
    version: {6,7}
        version of the glacier inventory
    rgi_dir : string
        directory where the glacier geometry files are located
    rgi_out_dir : string
        location where the glacier geometry files should be positioned

    Returns
    -------
    rgi_raster_paths : list
        locations and filenames of the glacier rasters

    See Also
    --------
    which_rgi_regions, create_rgi_raster

    Notes
    -----
    The following nomenclature is used:

    - lon : longitude, range=-180...+180, unit=degrees
    - lat : latitude, range=-90...+90, unit=degrees
    - RGI : Randolph glacier inventory
    - S2 : Sentinel-2
    - AOI : area of interest
    """
    rgi_dir = RGI_DIR_DEFAULT if rgi_dir is None else rgi_dir
    rgi_out_dir = rgi_dir if rgi_out_dir is None else rgi_out_dir

    # search in which region the glacier is situated, and download its
    # geometric data, if this in not present on the local machine
    rgi_paths = download_rgi(aoi, version, rgi_dir)

    if isinstance(aoi, str):
        mgrs_codes = [aoi]
    else:
        mgrs_codes = get_tile_codes_from_geom(aoi)

    rgi_raster_paths = []
    for mgrs_code in mgrs_codes:
        geoTransform, crs = get_generic_s2_raster(mgrs_code)
        rgi_raster_path = os.path.join(rgi_out_dir, f'{mgrs_code}.tif')
        rgi_shapes = _get_rgi_shapes(rgi_paths, version)
        create_rgi_raster(rgi_shapes, geoTransform, crs, rgi_raster_path)
        rgi_raster_paths.append(rgi_raster_path)
    return rgi_raster_paths
