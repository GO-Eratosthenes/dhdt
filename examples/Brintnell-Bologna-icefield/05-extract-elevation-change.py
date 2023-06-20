"""
Collect shadowcast data and merge this to spatialtemporal elevation change
"""
import glob
import os

import numpy as np
import pandas as pd

from dhdt.generic.debugging import start_pipeline
from dhdt.generic.mapping_io import read_geo_image
from dhdt.generic.mapping_tools import map2pix, ref_trans, ref_update
from dhdt.processing.coupling_tools import merge_ids_simple, \
    split_ids_view_angles
from dhdt.postprocessing.photohypsometric_tools import \
    read_conn_to_df, clean_locations_with_no_id, get_casted_elevation_difference, \
    get_hypsometric_elevation_change, update_casted_elevation, \
    clean_dxyt_via_common_glacier, get_mass_balance_per_elev, \
    write_df_to_belev_file

# initialization parameters
BBOX = [543001, 6868001, 570001, 6895001] # this is the way rasterio does it
MGRS_TILE = "09VWJ"

DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), 'data'))
CONN_DIR = os.path.join(DATA_DIR, "processing")
DUMP_DIR = os.path.join(DATA_DIR, "glaciers")
RGI_PATH = os.path.join(DATA_DIR, "RGI", MGRS_TILE+'.tif')
DEM_PATH = os.path.join(DATA_DIR, "DEM", MGRS_TILE+'.tif')

# import and create general assets
rgi_dat,_,rgi_aff,_ = read_geo_image(RGI_PATH)
dem_dat,crs,dem_aff,_ = read_geo_image(DEM_PATH)

# decrease the image size to bounding box
bbox_xy = np.array(BBOX).reshape((2,2)).T.ravel()
bbox_i, bbox_j = map2pix(rgi_aff, bbox_xy[0:2], np.flip(bbox_xy[2::]))
bbox_i, bbox_j = np.round(bbox_i).astype(int), np.round(bbox_j).astype(int)
new_aff = ref_trans(rgi_aff, int(bbox_i[0]), int(bbox_j[0]))
rgi_dat = rgi_dat[bbox_i[0]:bbox_i[1],bbox_j[0]:bbox_j[1]]
dem_dat = dem_dat[bbox_i[0]:bbox_i[1],bbox_j[0]:bbox_j[1]]
new_aff = ref_update(new_aff, rgi_dat.shape[0], rgi_dat.shape[1])


def main():
    # create structure
    os.makedirs(DUMP_DIR, exist_ok=True)

    # load data
    conn_files = glob.glob(os.path.join(CONN_DIR, '*/*/*/conn.txt'))
    dh = pd.concat([read_conn_to_df(file) for file in conn_files])

    # pair posts (caster locations)
    rgi_ids = np.unique(rgi_dat)
    rgi_ids = rgi_ids[rgi_ids != 0]
    for rgi in rgi_ids:
        dh_rgi = dh[dh['glacier_id'] == rgi]
        # prepare
        dxyt = (dh_rgi
                .pipe(start_pipeline)
                .pipe(merge_ids_simple)
                .pipe(split_ids_view_angles)
                .pipe(get_casted_elevation_difference)
                .pipe(update_casted_elevation, dem_dat, new_aff)
                )

        #processing
        dhdt = (dxyt
                .pipe(start_pipeline)
                .pipe(clean_dxyt_via_common_glacier)
                .pipe(get_hypsometric_elevation_change, dxyt)
                )
        del dxyt

        belev = get_mass_balance_per_elev(dhdt)
        write_df_to_belev_file(belev, DUMP_DIR, rgi_num=rgi, rgi_region=2)

if __name__ == "__main__":
    main()
