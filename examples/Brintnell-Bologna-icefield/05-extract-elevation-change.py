"""
Collect shadowcast data and merge this to spatialtemporal elevation change
"""
import os

from dhdt.generic.debugging import start_pipeline
from dhdt.generic.mapping_io import read_geo_image
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_files_to_df, clean_dh, get_casted_elevation_difference, \
    get_hypsometric_elevation_change, update_casted_elevation, \
    update_glacier_id, clean_dxyt

MGRS_TILE = "09VWJ"

DATA_DIR = os.path.join(os.getcwd(), "data") #os.sep.join(["project", "eratosthenes", "data"])
CONN_DIR = os.path.join(os.getcwd(), "processing") #os.sep.join(["project", "eratosthenes", "processing"])
DUMP_DIR = os.path.join(os.getcwd(), "glaciers")
RGI_PATH = os.path.join(DATA_DIR, "RGI", MGRS_TILE+'.tif')
DEM_PATH = os.path.join(DATA_DIR, "DEM", MGRS_TILE+'.tif')


def _get_stac_dates():
    stac_dates = []
    for root,directories,_ in os.walk(CONN_DIR):
        ymd = os.path.relpath(root, CONN_DIR)
        if ymd.count(os.sep)==2:
            stac_dates.append(ymd)
    return tuple(stac_dates)

def main():
    # loop through all different glaciers

    # load data
    im_paths = _get_stac_dates()
    dh = read_conn_files_to_df(im_paths, folder_path=CONN_DIR,
                               dist_thres=9.)
    rgi_dat,_,rgi_aff,_ = read_geo_image(RGI_PATH)
    dem_dat,_,dem_aff,_ = read_geo_image(DEM_PATH)

    # preprocessing
    dh = (dh
          .pipe(start_pipeline)
          .pipe(update_glacier_id, rgi_dat, rgi_aff)
          .pipe(clean_dh)
          )

    # prepare
    dxyt = (dh
            .pipe(start_pipeline)
            .pipe(get_casted_elevation_difference)
            .pipe(update_casted_elevation, dem_dat, dem_aff)
            )
    del dh

    #processing
    dhdt = (dxyt
            .pipe(start_pipeline)
            .pipe(clean_dxyt)
            .pipe(get_hypsometric_elevation_change, dxyt)
            )
    del dxyt

if __name__ == "__main__":
    main()
