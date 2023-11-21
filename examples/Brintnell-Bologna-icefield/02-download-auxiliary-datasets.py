"""
Download and set up the auxiliary datasets for a provided area of interest.
The following acronyms are used:
- Military Grid Reference System (MGRS) tiling schema.
- Randolph Glacier Inventory (RGI).
- Digital Elevation Model (DEM).
"""
import os

import numpy as np

from dhdt.auxiliary.handler_mgrs import download_mgrs_tiling
from dhdt.auxiliary.handler_randolph import create_rgi_tile_s2
from dhdt.auxiliary.handler_copernicusdem import make_copDEM_mgrs_tile
from dhdt.auxiliary.handler_era5 import get_era5_atmos_profile

from dhdt.generic.handler_stac import read_stac_catalog
from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import get_mean_map_location, map2pix, \
    ref_trans, ref_update

# Brintnell-Bologna icefield (Northwest Territories)
MGRS_TILE = "09VWJ"
BBOX = [543001, 6868001, 570001, 6895001]

ROOT_DIR = os.getenv("ROOT_DIR", os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, "data")
STAC_L1C_PATH = os.path.join(DATA_DIR, "SEN2", "sentinel2-l1c")

ITEM_ID = os.getenv("ITEM_ID", None)


def get_mean_coords_and_heights(dem, crs, transform):
    bbox_xy = np.array(BBOX).reshape((2, 2)).T.ravel()
    bbox_i, bbox_j = map2pix(transform, bbox_xy[0:2], np.flip(bbox_xy[2::]))
    bbox_i, bbox_j = np.round(bbox_i).astype(int), np.round(bbox_j).astype(int)
    new_aff = ref_trans(transform, int(bbox_i[0]), int(bbox_j[0]))
    dem_dat = dem[bbox_i[0]:bbox_i[1], bbox_j[0]:bbox_j[1]]
    new_aff = ref_update(new_aff, dem_dat.shape[0], dem_dat.shape[1])
    h = np.linspace(0, np.round(np.max(dem_dat) / 100) * 100 + 100, 10)
    x_bar, y_bar = get_mean_map_location(new_aff, crs)
    return x_bar, y_bar, h


def main():
    catalog = read_stac_catalog(STAC_L1C_PATH)

    if ITEM_ID is not None:
        items = [catalog.get_item(ITEM_ID, recursive=True)]
    else:
        items = catalog.get_all_items()

    mgrs_index = download_mgrs_tiling(
        tile_dir=os.path.join(DATA_DIR, "MGRS")
    )
    create_rgi_tile_s2(
        MGRS_TILE,
        rgi_dir=os.path.join(DATA_DIR, "RGI"),
        tile_path=mgrs_index,
    )
    dem_path = make_copDEM_mgrs_tile(
        MGRS_TILE,
        dem_dir=os.path.join(DATA_DIR, "DEM"),
        tile_path=mgrs_index,
    )

    dem, crs, transform, _ = read_geo_image(dem_path)
    x, y, h = get_mean_coords_and_heights(dem, crs, transform)
    for item in items:
        date = np.datetime64(item.datetime.strftime("%Y-%m-%d"))
        get_era5_atmos_profile(
            date, x, y, crs, h, os.path.join(DATA_DIR, "ERA5")
        )



if __name__ == "__main__":
    main()
