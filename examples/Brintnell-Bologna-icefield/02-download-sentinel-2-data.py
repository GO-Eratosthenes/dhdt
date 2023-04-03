"""
Search, download and create spatial temporal asset catalogue (STAC)
"""
import os
import datetime
import geojson

from sentinelsat import SentinelAPI

from dhdt.auxilary.handler_google_cloud import \
    create_stac_catalog, download_assets

MGRS_TILE = "09VWJ"
YOI = [2016, 2022]
DATA_DIR = os.path.join(os.getcwd(), 'data') # "/project/eratosthenes/Data/"

COPERNICUS_API_URL = "https://apihub.copernicus.eu/apihub"
COPERNICUS_HUB_USERNAME = os.getenv('COPERNICUS_HUB_USERNAME')
COPERNICUS_HUB_PASSWORD = os.getenv('COPERNICUS_HUB_PASSWORD')


AMAZON_API_URL = "https://earth-search.aws.element84.com/v0"
GCS_ROOT_URL = "https://storage.googleapis.com"
SENTINEL2_BUCKET = "gcp-public-data-sentinel-2"

collection_L1C = "sentinel-s2-l1c"
collection_L2A = "sentinel-s2-l2a-cogs"
asset_keys = "blue green red nir " +\
             "product_metadata granule_metadata inspire_metadata datastrip_metadata " +\
             "sensor_metadata_B02 sensor_metadata_B03 sensor_metadata_B04 sensor_metadata_B08"
asset_keys = asset_keys.split(" ")

def main():
    counter = 0

    # create selection criteria such as timespan and other specifics
    toi = (datetime.date(YOI[counter]+0, 10, 1),
           datetime.date(YOI[counter]+1, 4, 1))

    api = SentinelAPI(
        user=COPERNICUS_HUB_USERNAME,
        password=COPERNICUS_HUB_PASSWORD,
        api_url=COPERNICUS_API_URL,
        show_progressbars=False,
    )

    scenes_L1C = api.query(
        platformname="Sentinel-2",
        producttype="S2MSI1C",
        tileid=MGRS_TILE,
        date=toi,
        cloudcoverpercentage=(0, 70),
    )

    geojson_path = os.path.join(DATA_DIR, 'SEN2', 'sentinel2-l1c.json')
    catalog_path = os.path.dirname(geojson_path)
    os.makedirs(catalog_path, exist_ok=True)
    with open(geojson_path, "w") as f:
        feature_collection = api.to_geojson(scenes_L1C)
        geojson.dump(feature_collection, f)

    # download from google cloud storage
    catalog_description = "This catalog contains MGRS tiles that include " +\
        "the Brintnell-Bologna icefield (Rugged Range, Canada) as retrieved" +\
        "from the Google Cloud Storage STAC endpoint."
    create_stac_catalog(catalog_path, geojson_path, catalog_description)

    download_assets(catalog_path, asset_keys)

if __name__ == "__main__":
    main()