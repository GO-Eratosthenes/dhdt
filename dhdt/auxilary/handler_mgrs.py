import os
import geopandas as gpd
import re
import numpy as np
import mgrs
from shapely import wkt
from shapely.geometry import Polygon
from dhdt.generic.handler_www import get_file_from_www
from fiona.drvsupport import supported_drivers
supported_drivers['KML'] = 'rw'

"""
    Notes
    -----
    The tile structure is a follows "AABCC"
        * "AA" utm zone number, starting from the East, with steps of 8 degrees
        * "B" latitude zone, starting from the South, with steps of 6 degrees

    The following acronyms are used:

    - CRS : coordinate reference system
    - MGRS : US military grid reference system
    - s2 : Sentinel-2
    - WKT : well known text
"""

MGRS_TILING_URL = (
    "https://sentinels.copernicus.eu/documents/247904/1955685/"
    "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000"
    "_21000101T000000_B00.kml"
)

MGRS_TILING_DIR_DEFAULT = os.path.join('.', 'data', 'MGRS')


def download_mgrs_tiling(mgrs_dir=None, output='sentinel2_tiles_world.geojson', overwrite=False):
    """
    Retrieve mgrs tiling polygons. 

    Parameters
    ----------
    mgrs_dir : str, optional
        location where the MGRS tiling files will be saved. If None, files will be written to './data/MGRS'. By default None
    output: str, optional
        Output file name, by default 'sentinel2_tiles_world.geojson'
    overwrite: bool
        if True and file exists, download it again, otherwise do nothing
    Returns
    -------
    mgrs_path : str
        path to the extracted MGRS tiling file

    Notes
    -----
    KML file with the MGRS tiling scheme used by Sentinel-2 is provided by
    Copernicus [1]

    [1] https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2/data-products
    """

    mgrs_dir = MGRS_TILING_DIR_DEFAULT if mgrs_dir is None else mgrs_dir

    # download MGRS tiling file
    f_kml = get_file_from_www(MGRS_TILING_URL, mgrs_dir, overwrite)

    # Convert KML to geojson
    f_geojson = os.path.join(mgrs_dir, output)
    if not os.path.isfile(f_geojson) or overwrite:
        gdf = _kml_to_gdf(os.path.join(mgrs_dir, f_kml))
        gdf.to_file(f_geojson)
    return f_geojson


def _kml_to_gdf(filename):
    """
    Read MGRS kml file as a GeoPandas DataFrame, keep only polygon geometries.

    Parameters
    ----------
    filename : str
        kml file name

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        KML file read as a GeoDataFrame
    """
    # Load kml file
    gdf = gpd.read_file(filename, driver="KML")

    # Drop Description, whick is KML specific for visualization
    gdf = gdf.drop(columns=['Description'])

    # Unpack geometries
    gdf_exploded = gdf.explode(index_parts=False)

    # Select all entries which are Polygon
    mask = gdf_exploded["geometry"].apply(lambda x: isinstance(x, Polygon))
    gdf_out = gdf_exploded.loc[mask]

    return gdf_out


def get_bbox_from_tile_code(tile_code, geom_path=None, search_size=0.1):
    """  get the bounds of a certain MGRS tile

    Parameters
    ----------
    tile_code : string, e.g.: '05VMG'
        MGRS tile coding
    geom_path : string
        Path to the geometric metadata
    search_size : float
        Size of the search box, by default 0.1

    Returns
    -------
    bbox : numpy.ndarray, size=(1,4), dtype=float
        bounding box, in the following order: min max X, min max Y
    """
    if geom_path is None:
        geom_path = os.path.join(
            MGRS_TILING_DIR_DEFAULT, 'sentinel2_tiles_world.geojson')

    tile_code = _normalize_mgrs_code(tile_code)

    # Derive a search box from the tile code
    search_box = _mgrs_to_searchbox(tile_code, search_size)

    # Load tiles intersects the search box
    mgrs_tiles = gpd.read_file(geom_path, bbox=search_box)

    toi = mgrs_tiles[mgrs_tiles['Name'] == tile_code].total_bounds
    bbox = np.array([toi[0], toi[2], toi[1], toi[3]])
    return bbox


def get_geom_for_tile_code(tile_code, geom_path=None, search_size=0.1):
    """ get the geometry of a certain MGRS tile

    Parameters
    ----------
    tile_code : string, e.g.: '05VMG'
        MGRS tile coding
    geom_path : string
        Path to the geometric metadata
    search_size : float
        Size of the search box, by default 0.1

    Returns
    -------
    wkt : string
        well known text of the geometry, i.e.: 'POLYGON ((x y, x y, x y))'
    """

    if geom_path is None:
        geom_path = os.path.join(
            MGRS_TILING_DIR_DEFAULT, 'sentinel2_tiles_world.geojson')

    tile_code = _normalize_mgrs_code(tile_code)

    # Derive a search box from the tile code
    search_box = _mgrs_to_searchbox(tile_code, search_size)

    # Load tiles intersects the search box
    mgrs_tiles = gpd.read_file(geom_path, bbox=search_box)

    geom = mgrs_tiles[mgrs_tiles['Name'] == tile_code]["geometry"]

    if geom is None:
        print('MGRS tile code does not seem to exist')

    return geom


def get_tile_codes_from_geom(geom, geom_path=None):
    """
    Get the codes of the MGRS tiles intersecting a given geometry

    Parameters
    ----------
    geom : {shapely.geometry, string, dict, geopandas.GeoDataFrame, geopandas.GeoSeries}
        geometry object with the given dict-like geojson geometry, GeoSeries, GeoDataFrame or shapely geometry.
        or well known text, i.e.: 'POLYGON ((x y, x y, x y))'
    geom_path : string
        Path to the geometric metadata

    Returns
    -------
    tile_codes : tuple
        MGRS tile codes

    See Also
    --------
    .get_geom_for_tile_code, .get_bbox_from_tile_code
    """

    if geom_path is None:
        geom_path = os.path.join(
            MGRS_TILING_DIR_DEFAULT, 'sentinel2_tiles_world.geojson')

    # If a wkt str, convert to shapely geometry
    if isinstance(geom, str):
        try:
            geom = wkt.loads(geom)
        except Exception as e:
            raise e

    # Uniform CRS
    if isinstance(geom, gpd.GeoSeries) or isinstance(geom, gpd.GeoDataFrame):
        example = gpd.read_file(geom_path, rows=1)
        geom = geom.set_crs(example.crs)

    # Load tiles intersects the search box
    mgrs_tiles = gpd.read_file(geom_path, mask=geom)

    # Get the codes in tuple
    codes = tuple(mgrs_tiles["Name"])

    return codes


def _normalize_mgrs_code(tile_code):
    """
    Validate a MGRS tile code and make it uppercase

    Parameters
    ----------
    tile_code : str
        MGRS tile code

    Returns
    -------
    str
        validated and normalized tile code
    """
    # ToDo: raiser error instead of assert
    assert isinstance(tile_code, str), 'please provide a string'
    tile_code = tile_code.upper()
    assert bool(re.match("[0-9][0-9][A-Z][A-Z][A-Z]", tile_code)), \
        ('please provide a correct MGRS tile code')
    return tile_code


def _mgrs_to_searchbox(tile_code, size):
    """
    Get a squre search box from the tile code.

    Parameters
    ----------
    tile_code : str
        MGRS code of the tile
    size : float, optional
        size of the search box in degree

    Returns
    -------
    tuple
        boudingbox of the search area in (minx, miny, maxx, maxy)
    """

    m = mgrs.MGRS()
    clat, clon = m.toLatLon(tile_code)
    return (clon-size, clat-size, clon+size, clat+size)
