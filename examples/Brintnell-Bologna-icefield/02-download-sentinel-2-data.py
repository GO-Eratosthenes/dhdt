"""
Search, download and create spatial temporal asset catalogue (STAC)
"""
import os

import datetime
import pystac
import pystac_client


API_URL = "https://earth-search.aws.element84.com/v0"
DATA_DIR = os.path.join(os.getcwd(), 'data') # "/project/eratosthenes/Data/"

MGRS_TILE = "09VWJ"
YOI = [2016, 2022]

collection_L1C = "sentinel-s2-l1c"
collection_L2A = "sentinel-s2-l2a-cogs"


def main():
    counter = 0
    utm_zone = MGRS_TILE[0:2]
    latitude_band, grid_square = MGRS_TILE[2], MGRS_TILE[3:]

    # create selection criteria such as timespan and other specifics
    toi = [str(YOI[counter]+0) + '-10-01T00:00:00Z',
           str(YOI[counter]+1) + '-04-01T00:00:00Z']
    specifics = [
            f"sentinel:utm_zone={utm_zone}",
            f"sentinel:latitude_band={latitude_band}",
            f"sentinel:grid_square={grid_square}"
        ]

    client = pystac_client.Client.open(API_URL)

    search_L1C = client.search(
        collections=[collection_L1C], datetime=toi, query=specifics
        )
    items_L1C = search_L1C.get_all_items()

    search_L2A = client.search(
        collections=[collection_L2A], datetime=toi, query=specifics
        )
    items_L2A = search_L2A.get_all_items()

    # find scenes that are not present in both repositories
    L1C_dates = [item.properties["datetime"] for item in items_L1C]
    missing = [item for item in items_L2A if item.properties["datetime"] not in L1C_dates]

    # create STACcatalogue
    catalog_id = "SEN2"
    catalog = pystac.Catalog(
        id=catalog_id,
        description=("This catalog contains MGRS tiles that include "
                     "the Brintnell-Bologna icefield (Rugged Range, "
                     "Canada) as retrieved from the Earth Search "
                     "STAC endpoint.")
    )
    # add search results to catalog
    for items in (items_L1C, items_L2A):
        catalog.add_items(items)
    # replace the self-links of the items to remote with relative links
    catalog.normalize_hrefs(catalog_id)

    template = "${collection}/${year}/${month}/${day}"
    catalog.generate_subcatalogs(template)
    catalog.normalize_and_save(
        os.path.join(DATA_DIR, catalog_id),
        catalog_type='SELF_CONTAINED'
    )
