"""
Generate a SpatioTemporal Asset Catalog (STAC) from the Sentinel-2 metadata
returned by the Copernicus Open Access Hub API. Assets are linked to files on
the Sentinel-2 public dataset on Google Cloud Storage (GCS):

https://cloud.google.com/storage/docs/public-datasets/sentinel-2

"""
import argparse
import os

from typing import Dict, Tuple

import geopandas as gpd
import pystac

from stactools.core.io.xml import XmlElement
from stactools.sentinel2.stac import create_item
from stactools.sentinel2.constants import GRANULE_METADATA_ASSET_KEY


GCS_ROOT_URL = "https://storage.googleapis.com"
SENTINEL2_BUCKET = "gcp-public-data-sentinel-2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "geojson", type=str, help="Path to the Sentinel-2 metadata file"
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the STAC catalog",
        default="./catalog",
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Brief catalog description",
        default="My STAC catalog",
    )
    parser.add_argument(
        "--template", type=str, help="Catalog structure",
    )
    return parser.parse_args()


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
    return f"sensor-metadata-{band_key}", asset


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


def create_stac_catalog(
        geojson_path: str, catalog_path: str, catalog_description: str,
        template: str = None
) -> None:
    """
    Create a STAC catalog from the scenes' metadata retrieved from the
    Copernicus Open Access Hub API and saved locally in GeoJSON format. Assets
    are linked to the corresponding files on the Sentinel-2 public dataset on
    Google Cloud Storage.

    Parameters
    ----------
    geojson_path: str
        path to the GeoJSON file containing the scenes' metadata
    catalog_path: str
        path where to save the STAC catalog
    catalog_description: str
        short description of the catalog
    template: str, optional
        if provided, organize items in a nested catalog structure according to
        this template
    Returns
    -------
    pystac.Catalog : list of items corresponding to the Sentinel-2 scenes.
    """

    _, catalog_id = os.path.split(catalog_path)
    catalog = pystac.Catalog(id=catalog_id, description=catalog_description)

    scenes_df = gpd.read_file(geojson_path)
    items = (_create_item(scene.filename) for _, scene in scenes_df.iterrows())
    catalog.add_items(items)

    if template is not None:
        catalog.generate_subcatalogs(template)

    catalog.normalize_and_save(
        catalog_path,
        catalog_type=pystac.CatalogType.SELF_CONTAINED
    )


def main():
    args = parse_args()
    create_stac_catalog(
        geojson_path=args.geojson,
        catalog_path=args.path,
        catalog_description=args.description,
        template=args.template,
    )


if __name__ == "__main__":
    main()
