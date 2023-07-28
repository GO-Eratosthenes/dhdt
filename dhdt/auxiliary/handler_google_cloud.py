import os

import pystac
from stactools.core.copy import move_asset_file_to_item
from stactools.core.io.xml import XmlElement
from stactools.sentinel2.stac import create_item
from stactools.sentinel2.constants import GRANULE_METADATA_ASSET_KEY

GCS_ROOT_URL = "https://storage.googleapis.com"
SENTINEL2_BUCKET = "gcp-public-data-sentinel-2"


def _get_granule_href_s2(safe_filename):
    """ Construct HREF with full GCS URL from the SAFE file name.

    Parameters
    ----------
    safe_filename : string
        .SAFE-filename of the associated Sentinel-2 scene of interest

    Returns
    -------
    string
        full address in the google cloud storage
    """
    bucket_url = f"{GCS_ROOT_URL}/{SENTINEL2_BUCKET}"
    root_url = (bucket_url
                if "_MSIL1C_" in safe_filename else f"{bucket_url}/L2")
    tile_id = safe_filename[39:44]
    return (f"{root_url}/"
            f"tiles/{tile_id[:2]}/{tile_id[2]}/{tile_id[3:]}/"
            f"{safe_filename}")


def _get_sensor_array_key_asset_pair_s2(href):
    """
    Create the asset corresponding to a sensor array data file from its HREF.

    Parameters
    ----------
    href : string
        location of the meta-data file

    Returns
    -------
    Tuple[str, pystac.Asset]

    Notes
    -----
    The GML format was employed for older Sentinel-2 scenes, while JPEG2000 is
    used for the most recent ones.
    """
    stem, ext = os.path.splitext(href)
    band_key = stem.split("_")[-1]
    media_type = (pystac.MediaType.XML
                  if ext == ".gml" else pystac.MediaType.JPEG2000)
    asset = pystac.Asset(href=href, media_type=media_type, roles=["metadata"])
    return f"sensor_metadata_{band_key}", asset


def _create_sensor_array_assets_s2(granule_href, granule_metadata_href):
    """
    Set up an asset dictionary including the sensor array data. Extract names
    and relative paths of the data files from the granule metadata file.

    Parameters
    ----------
    granule_href: string
        .SAFE-filename of the associated Sentinel-2 scene of interest
    granule_metadata_href: string
        .SAFE-filename of the associated Sentinel-2 scene of interest

    Returns
    -------
    Dict[str, pystac.Asset]
    """
    sensor_array_assets = {}
    root = XmlElement.from_file(granule_metadata_href)
    elements = (root.findall(
        "n1:Quality_Indicators_Info/Pixel_Level_QI/MASK_FILENAME"))
    for element in elements:
        if element.get_attr("type") == "MSK_DETFOO":
            asset_href = os.path.join(granule_href, element.text)
            key, asset = _get_sensor_array_key_asset_pair_s2(asset_href)
            sensor_array_assets[key] = asset
    return sensor_array_assets


def _create_item_s2(safe_filename):
    """ Create a STAC item corresponding to a SAFE filename.

    Parameters
    ----------
    safe_filename : string
        .SAFE-filename of the associated Sentinel-2 scene of interest

    Returns
    -------
    pystac.Item
    """
    granule_href = _get_granule_href_s2(safe_filename)
    item = create_item(granule_href)
    granule_metadata_href = item.assets[GRANULE_METADATA_ASSET_KEY].href
    sensor_array_assets = _create_sensor_array_assets_s2(
        granule_href, granule_metadata_href)
    item.assets.update(sensor_array_assets)
    return item


def create_stac_catalog(catalog_path,
                        catalog_description,
                        safe_filenames,
                        template='${year}/${month}/${day}'):
    """
    Create a STAC catalog from the provided list of Sentinel-2 scenes,
    as specified via the granule SAFE filenames. Assets are linked to the
    corresponding files on the Sentinel-2 public dataset on Google Cloud
    Storage.
    """
    _, catalog_id = os.path.split(catalog_path)
    catalog = pystac.Catalog(id=catalog_id, description=catalog_description)

    for safe_filename in safe_filenames:
        catalog.add_item(_create_item_s2(safe_filename))

    if template is not None:
        catalog.generate_subcatalogs(template)

    catalog.normalize_and_save(catalog_path,
                               catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return catalog


def download_assets(catalog_path, asset_keys):
    """
    Download (a subsection of) the catalog assets and store these alongside the
    corresponding items.

    Parameters
    ----------
    catalog_path : string
        location where the json-file is located

    """
    path = catalog_path \
        if catalog_path.endswith("catalog.json") \
        else f"{catalog_path}/catalog.json"
    catalog = pystac.Catalog.from_file(path)

    for item in catalog.get_all_items():
        keys = asset_keys if asset_keys is not None else item.assets.keys()
        for key in keys:
            asset = item.assets[key]
            asset_abs_href = asset.get_absolute_href()
            asset_href = move_asset_file_to_item(
                item,
                asset_abs_href,
                copy=True,
            )
            asset.href = pystac.utils.make_relative_href(
                asset_href, item.get_self_href())
        item.save_object()
