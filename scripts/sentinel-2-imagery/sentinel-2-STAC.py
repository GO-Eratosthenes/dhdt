"""
Generate a SpatioTemporal Asset Catalog (STAC) from the Sentinel-2 metadata
returned by the Copernicus Open Access Hub API. Assets are linked to files on
the Sentinel-2 public dataset on Google Cloud Storage (GCS)[1]. Optionally
download (a subset of) the assets and store these alongside the respective
scenes' metadata.

https://cloud.google.com/storage/docs/public-datasets/sentinel-2

"""
__version__ = "0.1"

import argparse
import os

from typing import Dict, Tuple

import geopandas as gpd
import pystac

from pystac.utils import make_relative_href
from stactools.core.copy import move_asset_file_to_item
from stactools.core.io.xml import XmlElement
from stactools.sentinel2.stac import create_item
from stactools.sentinel2.constants import GRANULE_METADATA_ASSET_KEY


GCS_ROOT_URL = "https://storage.googleapis.com"
SENTINEL2_BUCKET = "gcp-public-data-sentinel-2"


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog-path",
        type=str,
        help="Path to the STAC catalog",
        required=True,
    )
    subparsers = parser.add_subparsers(required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument(
        "--from-geojson",
        type=str,
        help="Path to the Sentinel-2 metadata file",
        required=True,
    )
    create_parser.add_argument(
        "--description",
        type=str,
        help="Brief catalog description",
        default="My STAC catalog",
        required=True,
    )
    create_parser.add_argument(
        "--template", type=str, help="Catalog structure",
    )
    create_parser.set_defaults(run=create_stac_catalog)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument(
        "--assets", nargs='*', help="List of asset keys"
    )
    download_parser.set_defaults(run=download_assets)
    return parser.parse_args(args)


def _get_granule_href(safe_filename: str) -> str:
    """ Construct HREF with full GCS URL from the SAFE file name. """
    bucket_url = f"{GCS_ROOT_URL}/{SENTINEL2_BUCKET}"
    root_url = (
        bucket_url if "_MSIL1C_" in safe_filename else f"{bucket_url}/L2"
    )
    tile_id = safe_filename[39:44]
    return (
        f"{root_url}/"
        f"tiles/{tile_id[:2]}/{tile_id[2]}/{tile_id[3:]}/"
        f"{safe_filename}"
    )


def _get_sensor_array_key_asset_pair(href: str) -> Tuple[str, pystac.Asset]:
    """
    Create the asset corresponding to a sensor array data file from it HREF.

    Notes
    -----
    The GML format was employed for older Sentinel-2 scenes, while JPEG2000 is
    used for the most recent ones.
    """
    stem, ext = os.path.splitext(href)
    band_key = stem.split("_")[-1]
    media_type = (
        pystac.MediaType.XML if ext == ".gml" else pystac.MediaType.JPEG2000
    )
    asset = pystac.Asset(href=href, media_type=media_type, roles=["metadata"])
    return f"sensor_metadata_{band_key}", asset


def _create_sensor_array_assets(
        granule_href: str, granule_metadata_href: str
) -> Dict[str, pystac.Asset]:
    """
    Set up an asset dictionary including the sensor array data. Extract names
    and relative paths of the data files from the granule metadata file.
    """
    sensor_array_assets = {}
    root = XmlElement.from_file(granule_metadata_href)
    elements = (
        root.findall("n1:Quality_Indicators_Info/Pixel_Level_QI/MASK_FILENAME")
    )
    for element in elements:
        if element.get_attr("type") == "MSK_DETFOO":
            asset_href = os.path.join(granule_href, element.text)
            key, asset = _get_sensor_array_key_asset_pair(asset_href)
            sensor_array_assets[key] = asset
    return sensor_array_assets


def _create_item(safe_filename: str) -> pystac.Item:
    """ Create a STAC item corresponding to a SAFE filename. """
    granule_href = _get_granule_href(safe_filename)
    item = create_item(granule_href)
    granule_metadata_href = item.assets[GRANULE_METADATA_ASSET_KEY].href
    sensor_array_assets = _create_sensor_array_assets(
        granule_href, granule_metadata_href
    )
    item.assets.update(sensor_array_assets)
    return item


def create_stac_catalog(args: argparse.Namespace) -> None:
    """
    Create a STAC catalog from the scenes' metadata retrieved from the
    Copernicus Open Access Hub API and saved locally in GeoJSON format. Assets
    are linked to the corresponding files on the Sentinel-2 public dataset on
    Google Cloud Storage.
    """
    catalog_path = args.catalog_path
    geojson_path = args.from_geojson
    catalog_description = args.description
    template = args.template

    _, catalog_id = os.path.split(catalog_path)
    catalog = pystac.Catalog(id=catalog_id, description=catalog_description)

    scenes_df = gpd.read_file(geojson_path)
    items = (_create_item(scene.filename) for _, scene in scenes_df.loc[:2].iterrows())
    catalog.add_items(items)

    if template is not None:
        catalog.generate_subcatalogs(template)

    catalog.normalize_and_save(
        catalog_path,
        catalog_type=pystac.CatalogType.SELF_CONTAINED
    )


def download_assets(args: argparse.Namespace) -> None:
    """
    Download (a subsection of) the catalog assets and store these alongside the
    corresponding items.
    """
    catalog_path = args.catalog_path
    asset_keys = args.assets

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
            asset.href = make_relative_href(asset_href, item.get_self_href())
        item.save_object()


def main(args: list = None) -> None:
    args_parsed = parse_args(args)
    args_parsed.run(args_parsed)


if __name__ == "__main__":
    main()
