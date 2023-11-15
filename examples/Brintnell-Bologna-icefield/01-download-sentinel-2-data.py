"""
Search and download Sentinel-2 data as a SpatioTemporal Asset Catalog (STAC)
"""
import os
import datetime

from sentinelsat import SentinelAPI

from dhdt.auxilary.handler_google_cloud import \
    create_stac_catalog, download_assets

MGRS_TILE = "09VWJ"
ROOT_DIR = os.getenv("ROOT_DIR", os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, "data")
STAC_L1C_PATH = os.path.join(DATA_DIR, "SEN2", "sentinel2-l1c")
STAC_L2A_PATH = os.path.join(DATA_DIR, "SEN2", "sentinel2-l2a")
STAC_DESCRIPTION = (
    "Sentinel-2 data catalog containing MGRS tiles that include the ",
    "Brintnell-Bologna icefield (Rugged Range, Canada) as retrieved ",
    "from the Google Cloud Storage",
)

COPERNICUS_API_URL = "https://apihub.copernicus.eu/apihub"
COPERNICUS_HUB_USERNAME = os.getenv('COPERNICUS_HUB_USERNAME')
COPERNICUS_HUB_PASSWORD = os.getenv('COPERNICUS_HUB_PASSWORD')

L1C_ASSET_KEYS = [
    "blue", "green", "red", "nir", "product_metadata", "granule_metadata",
    "inspire_metadata", "datastrip_metadata", "sensor_metadata_B02",
    "sensor_metadata_B03", "sensor_metadata_B04", "sensor_metadata_B08",
]
L2A_ASSET_KEYS = ["scl"]


def _get_matching_L2A_granules(scenes_L1C, scenes_L2A):
    # for older records, there is no L2A data
    if len(scenes_L2A) == 0:
        return []

    # match scene records
    merged = scenes_L1C.merge(
        scenes_L2A.dropna(subset="level1cpdiidentifier"),
        how="left",
        on="level1cpdiidentifier",
        validate="1:1",
        suffixes=["_L1C", "_L2A"]
    )

    # extract IDs of the L2A scenes matching L1C ones
    return merged["filename_L2A"].dropna().to_list()


def main():
    
    api = SentinelAPI(
        user=COPERNICUS_HUB_USERNAME,
        password=COPERNICUS_HUB_PASSWORD,
        api_url=COPERNICUS_API_URL,
        show_progressbars=False,
    )

    scenes_L1C = api.query(
        platformname="Sentinel-2",
        producttype="S2MSI1C",
        raw=f'filename:*_T{MGRS_TILE}_*',
        cloudcoverpercentage=(0, 70),
    )
    scenes_L1C = api.to_geodataframe(scenes_L1C)

    scenes_L2A = api.query(
        platformname="Sentinel-2",
        producttype="S2MSI2A",
        raw=f'filename:*_T{MGRS_TILE}_*',
    )
    scenes_L2A = api.to_geodataframe(scenes_L2A)

    granules_l1c = scenes_L1C["filename"].to_list()
    granules_l2a = _get_matching_L2A_granules(scenes_L1C, scenes_L2A)

    # download from Google Cloud Storage
    create_stac_catalog(STAC_L1C_PATH, STAC_DESCRIPTION, granules_l1c)
    create_stac_catalog(STAC_L2A_PATH, STAC_DESCRIPTION, granules_l2a)

    download_assets(STAC_L1C_PATH, L1C_ASSET_KEYS)
    download_assets(STAC_L2A_PATH, L2A_ASSET_KEYS)


if __name__ == "__main__":
    main()

