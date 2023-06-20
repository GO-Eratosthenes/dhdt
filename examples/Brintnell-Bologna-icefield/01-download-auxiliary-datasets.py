"""
Download and set up the auxiliary datasets for a provided area of interest:
- Military Grid Reference System (MGRS) tiling schema.
- Randolph Glacier Inventory (RGI).
- Digital Elevation Model (DEM).
"""
import os

from dhdt.auxilary.handler_mgrs import download_mgrs_tiling
from dhdt.auxilary.handler_randolph import create_rgi_tile_s2
from dhdt.auxilary.handler_copernicusdem import make_copDEM_mgrs_tile


# Brintnell-Bologna icefield (Northwest Territories)
MGRS_TILE = "09VWJ"

ROOT_DIR = os.getenv("ROOT_DIR", os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, "data")


def main():
    mgrs_index = download_mgrs_tiling(
        tile_dir=os.path.join(DATA_DIR, "MGRS")
    )
    create_rgi_tile_s2(
        MGRS_TILE,
        rgi_dir=os.path.join(DATA_DIR, "RGI"),
        tile_path=mgrs_index,
    )
    make_copDEM_mgrs_tile(
        MGRS_TILE,
        dem_dir=os.path.join(DATA_DIR, "DEM"),
        tile_path=mgrs_index,
    )


if __name__ == "__main__":
    main()
