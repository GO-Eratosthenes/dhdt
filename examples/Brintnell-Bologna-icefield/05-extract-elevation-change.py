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

DATA_DIR = "/project/eratosthenes/Data/"
CONN_DIR = "/project/eratosthenes/Conn/"
RGI_PATH = os.path.join(DATA_DIR, "RGI", MGRS_TILE+'.tif')
DEM_PATH = os.path.join(DATA_DIR, "DEM", MGRS_TILE+'.tif')

im_path = ('S2_2017-10-20', 'S2_2020-10-19')

def main():
    # load data
    dh = read_conn_files_to_df(im_path, folder_path=CONN_DIR,
                               dist_thres=100.)
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
