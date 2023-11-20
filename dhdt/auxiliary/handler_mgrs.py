"""
The MGRS tile structure is a follows "AABCC"
    * "AA" utm zone number, starting from the East, with steps of 8 degrees
    * "B" latitude zone, starting from the South, with steps of 6 degrees
    * "CC" 100 km square identifier

The following acronyms are used:

- CRS : coordinate reference system
- MGRS : US military grid reference system
- WKT : well known text
"""
import os

import geopandas as gpd
import numpy as np
from fiona.drvsupport import supported_drivers
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon

from dhdt.generic.handler_www import get_file_from_www
from dhdt.generic.unit_check import check_mgrs_code

supported_drivers['KML'] = 'rw'

MGRS_TILING_URL = ("https://sentinels.copernicus.eu/documents/247904/1955685/"
                   "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000"
                   "_21000101T000000_B00.kml")

MGRS_TILING_FILENAME = 'sentinel2_tiles_world.geojson'
MGRS_TILING_DIR_DEFAULT = os.path.join('.', 'data', 'MGRS')


def download_mgrs_tiling(tile_dir=None, tile_file=None, overwrite=False):
    """
    Retrieve MGRS tiling polygons, and extract geometries to a vector file (by
    default GeoJSON)

    Parameters
    ----------
    tile_dir : str, optional
        location where the MGRS tiling files will be saved. If None, files will
        be written to './data/MGRS'. By default None
    tile_file: str, optional
        Output file name, by default 'sentinel2_tiles_world.geojson'
    overwrite: bool
        if True and file exists, download it again, otherwise do nothing

    Returns
    -------
    str
        path to the extracted MGRS tiling file

    Notes
    -----
    The KML file with the MGRS tiling scheme used by Sentinel-2 is provided by
    Copernicus [1]_.

    References
    ----------
    .. [1] https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-2/data-products
    """  # noqa: E501

    tile_dir = MGRS_TILING_DIR_DEFAULT if tile_dir is None else tile_dir
    tile_file = MGRS_TILING_FILENAME if tile_file is None else tile_file

    # download MGRS tiling file
    f_kml = get_file_from_www(MGRS_TILING_URL, tile_dir, overwrite)

    # Extract geometries from KML
    tile_path = os.path.join(tile_dir, tile_file)
    if not os.path.isfile(tile_path) or overwrite:
        gdf = _kml_to_gdf(os.path.join(tile_dir, f_kml))
        gdf.to_file(tile_path)
    return tile_path


def _kml_to_gdf(tile_path):
    """
    Read MGRS kml file as a GeoPandas DataFrame, keep only polygon geometries.

    Parameters
    ----------
    tile_path : str
        kml file name

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        KML file read as a GeoDataFrame
    """
    # Load kml file
    gdf = gpd.read_file(tile_path, driver="KML")

    # Drop Description, whick is KML specific for visualization
    gdf = gdf.drop(columns=['Description'])

    # Unpack geometries
    gdf_exploded = gdf.explode(index_parts=False)

    # Select all entries which are Polygon
    mask = gdf_exploded["geometry"].apply(lambda x: isinstance(x, Polygon))
    gdf_out = gdf_exploded.loc[mask]

    return gdf_out


def get_geom_for_tile_code(tile_code, tile_path=None):
    """
    Get the geometry of a certain MGRS tile

    Parameters
    ----------
    tile_code : string
        MGRS tile coding, e.g.: '05VMG'
    tile_path : string
        Path to the geometric metadata

    Returns
    -------
    shapely.geometry.polygon.Polygon
        Geometry of the MGRS tile, in lat/lon
    """
    if tile_path is None:
        tile_path = os.path.join(MGRS_TILING_DIR_DEFAULT, MGRS_TILING_FILENAME)

    tile_code = check_mgrs_code(tile_code)

    # Derive a search geometry from the tile code
    search_geom = _mgrs_to_search_geometry(tile_code)

    # Load tiles
    mgrs_tiles = gpd.read_file(tile_path, mask=search_geom)

    geom = mgrs_tiles[mgrs_tiles['Name'] == tile_code]["geometry"]

    if len(geom) == 0:
        raise ValueError('MGRS tile code does not seem to exist')

    return geom.unary_union


def get_bbox_from_tile_code(tile_code, tile_path=None):
    """
    Get the bounds of a certain MGRS tile

    Parameters
    ----------
    tile_code : string, e.g.: '05VMG'
        MGRS tile coding
    tile_path : string
        Path to the geometric metadata

    Returns
    -------
    numpy.ndarray, size=(1,4), dtype=float
        bounding box, in the following order: min max X, min max Y
    """

    geom = get_geom_for_tile_code(tile_code, tile_path=tile_path)

    toi = geom.bounds
    bbox = np.array([toi[0], toi[2], toi[1], toi[3]])
    return bbox


def get_tile_codes_from_geom(geom, tile_path=None):
    """
    Get the codes of the MGRS tiles intersecting a given geometry

    Parameters
    ----------
    geom : {shapely.geometry, string, dict, GeoDataFrame, GeoSeries}
        geometry object with the given dict-like geojson geometry, GeoSeries,
        GeoDataFrame, shapely geometry or well known text, i.e.:
        'POLYGON ((x y, x y, x y))'
    tile_path : string
        Path to the geometric metadata

    Returns
    -------
    tuple
        MGRS tile codes

    See Also
    --------
    .get_geom_for_tile_code, .get_bbox_from_tile_code
    """

    if tile_path is None:
        tile_path = os.path.join(MGRS_TILING_DIR_DEFAULT, MGRS_TILING_FILENAME)

    # If a wkt str, convert to shapely geometry
    if isinstance(geom, str):
        geom = wkt.loads(geom)

    # Uniform CRS
    if isinstance(geom, gpd.GeoSeries) or isinstance(geom, gpd.GeoDataFrame):
        example = gpd.read_file(tile_path, rows=1)
        geom = geom.set_crs(example.crs)

    # Load tiles intersects the search box
    tiles = gpd.read_file(tile_path, mask=geom)

    # Get the codes in tuple
    codes = tuple(tiles["Name"])

    return codes


def _mgrs_to_search_geometry(tile_code):
    """
    Get a geometry from the tile code. The search geometry is a 6-deg longitude
    stripe, augmented with the stripe across the antimeridian for the tiles
    that are crossing this line.

    Parameters
    ----------
    tile_code : str
        MGRS code of the tile

    Returns
    -------
    shapely.Geometry
        geometry to restrict the search area
    """
    nr_lon = int(tile_code[0:2])  # first two letters indicates longitude range
    min_lon = -180. + (nr_lon - 1) * 6.
    max_lon = -180. + nr_lon * 6.
    geom = Polygon.from_bounds(min_lon, -90.0, max_lon, 90.0)
    extra = None
    if nr_lon == 1:
        extra = Polygon.from_bounds(174, -90.0, 180, 90.0)
    elif nr_lon == 60:
        extra = Polygon.from_bounds(-180, -90.0, -174, 90.0)
    if extra is not None:
        geom = MultiPolygon(polygons=[geom, extra])
    return geom
