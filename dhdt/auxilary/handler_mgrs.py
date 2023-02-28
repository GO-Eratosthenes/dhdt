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


def download_mgrs_tiling(mgrs_dir=None, output='sentinel2_tiles_world.geojson', overwrite=False):
    """
    Retrieve mgrs tiling polygons. 

    Parameters
    ----------
    mgrs_dir : str, optional
        location where the MGRS tiling files will be saved. If None, files will be written to './data/MGRS'. By default None
    output: str, optional
        Output file name, by default 'sentinel2_tiles_world.geojson'

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

    # Unpack geometries
    gdf_exploded = gdf.explode(index_parts=False)

    # Select all entries which are Polygon
    mask = gdf_exploded["geometry"].apply(lambda x: isinstance(x, Polygon))
    gdf_out = gdf_exploded.loc[mask]

    return gdf_out
