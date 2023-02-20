import os

from dhdt.generic.handler_www import get_file_from_www


MGRS_TILING_URL = (
    "https://sentinels.copernicus.eu/documents/247904/1955685/"
    "S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000"
    "_21000101T000000_B00.kml"
)

MGRS_TILING_DIR_DEFAULT = os.path.join('.', 'data', 'MGRS')


def download_mgrs_tiling(mgrs_dir=None):
    """


    Parameters
    ----------
    mgrs_dir : str
        location where the MGRS tiling files will be saved

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
    get_file_from_www(MGRS_TILING_URL, mgrs_dir)

    convert_mgrs_tiling_to_geojson()


def convert_mgrs_tiling_to_geojson():
    pass
