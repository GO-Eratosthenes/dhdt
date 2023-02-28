import os
import geopandas as gpd
from shapely.geometry import Polygon
from dhdt.generic.handler_www import get_file_from_www
from fiona.drvsupport import supported_drivers
import pandas as pd
supported_drivers['KML'] = 'rw'

MGRS_TILING_URL = (
    "https://sentinels.copernicus.eu/documents/247904/1955685/"
    "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000"
    "_21000101T000000_B00.kml"
)

MGRS_TILING_DIR_DEFAULT = os.path.join('.', 'data', 'MGRS')


def download_mgrs_tiling(mgrs_dir=None, output='sentinel2_tiles_world.geojson', keep_kml=False):
    """
    Retrieve mgrs tiling polygons. 

    Parameters
    ----------
    mgrs_dir : str, optional
        location where the MGRS tiling files will be saved. If None, files will be written to './data/MGRS'. By default None
    output: str, optional
        Output file name, by default 'sentinel2_tiles_world.geojson'
    keep_kml : bool, optional
        If keep the kml format. If False, will convert the kml data to geojson. By default False

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
    f_kml = get_file_from_www(MGRS_TILING_URL, mgrs_dir)

    if not keep_kml:
        f_geojson = os.path.join(mgrs_dir, output)
        gdf = _kml_to_gdf(os.path.join(mgrs_dir, f_kml))
        gdf.to_file(f_geojson)
        return f_geojson
    else:
        return f_kml


def _kml_to_gdf(filename):
    """
    Read MGRS kml file as a GeoPandas DataFrame

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

    # Separate the entries crossing the antimeridian
    mask_cross = gdf["geometry"].apply(_check_cross)
    gdf_cross = gdf.loc[mask_cross].copy()
    gdf_normal = gdf.loc[~mask_cross].copy()

    # Read in normal polygons
    # Their geometry will contain two elements, the first one always footprint polygon
    # The second one is likely to be the centroid, which will be disgarded
    gdf_normal.update(gdf_normal["geometry"].apply(lambda x: x.geoms[0]))
    gdf_normal = gdf_normal.set_crs(gdf.crs)

    # Read polygons crossing the crossing the antimeridian
    # Their geometry will contain the three elements
    # The second or third element will be another polygon, which will be read as a separate row
    unpacked = gdf_cross.apply(_unpack_mgrs_geom, axis=1, result_type='expand')
    gs_cross_list = pd.concat([unpacked[idx]
                              for idx in unpacked.columns]).to_list()
    gdf_cross = gpd.GeoDataFrame(gs_cross_list)
    gdf_cross = gdf_cross.set_crs(gdf.crs)

    gdf = pd.concat([gdf_cross, gdf_normal])

    return gdf


def _check_cross(gs):
    """
    Check if a GeoSeries cross the Antimeridian. The ones crossing will contain more than one Polygons.


    Parameters
    ----------
    gs : geopandas.GeoSeries
        Input geoseries.

    Returns
    -------
    pandas.Series
        Mask series. Returns True if an entry cross the Antimeridian.
    """
    flag = False
    if len(gs.geoms) > 2:
        flag = True
    return flag


def _unpack_mgrs_geom(gs):
    """
    Unpack the Geometry Collection for entries crossing the Antimeridian to separate rows. 

    Parameters
    ----------
    gs : geopandas.GeoSeries
        Input geoseries with all entries crossing the Antimeridian.

    Returns
    -------
    List of geopandas.GeoSeries
        List of the unpacked geoseries, where each element contains a single polygon.
    """
    gs_list = []
    for geom in gs['geometry'].geoms:
        gs_copy = gs.copy()
        if isinstance(geom, Polygon):
            gs_copy['geometry'] = geom
            gs_list.append(gs_copy)

    return gs_list
