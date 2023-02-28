"""
Download and set up the auxiliary datasets for a provided area of interest:
- Military Grid Reference System (MGRS) tiling schema.
- Randolph Glacier Inventory (RGI).
- Digital Elevation Model (DEM).
- coast lines.
"""
import os

from dhdt.auxilary.handler_mgrs import download_mgrs_tiling
from dhdt.auxilary.handler_randolph import create_rgi_tile_s2
from dhdt.auxilary.handler_copernicusdem import make_copDEM_mgrs_tile


# Brintnell-Bologna icefield (Northwest Territories)
LAT = 62.09862204
LON = -127.9693738

DATA_DIR = "./data"


def main():
    aoi = (LON, LAT)
    mgrs_index = download_mgrs_tiling(
        mgrs_dir=os.path.join(DATA_DIR, "MGRS")
    )
    create_rgi_tile_s2(
        aoi=aoi,
        rgi_dir=os.path.join(DATA_DIR, "RGI"),
        mgrs_index=mgrs_index,
    )
    make_copDEM_mgrs_tile(
        aoi=aoi,
        dem_path=os.path.join(DATA_DIR, "DEM"),
        mgrs_index=mgrs_index,
    )


if __name__ == "__main__":
    main()
