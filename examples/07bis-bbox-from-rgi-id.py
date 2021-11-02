import os
import getpass

import numpy as np

import pystac
import stac2dcache

from eratosthenes.generic.mapping_io import read_geo_image

rgi_index_url = (
    "https://webdav.grid.surfsara.nl:2880"
    "/pnfs/grid.sara.nl/data/eratosthenes/"
    "disk/RasterRGI_tiles_sentinel-2"
    )

tile_id = '5VMG'
rgi_id = 'RGI60-01.19773' # Red Glacier
buffer_bbox = 1000 # [meter] extend the bbox a bit

# configure connection to dCache
username = getpass.getuser()
if username=='Alten005':
    dcache = stac2dcache.configure(
        filesystem="dcache", 
        token_filename=os.path.expanduser("~/Eratosthenes/HPC/macaroon.dat")
        )
else:
    dcache = stac2dcache.configure(
        filesystem="dcache", 
        token_filename="macaroon.dat"
        )

# download assets - from web to storage
dcache.download(os.path.join(rgi_index_url, tile_id+'.tif'), 
                tile_id+'.tif')

# read DEM index
RGI, spatialRef, geoTransform, targetprj = read_geo_image(tile_id+'.tif')
rgi_num = int(rgi_id.split('.')[1])

RGI_oi = np.where(RGI==rgi_num)
bbox = (np.min(RGI_oi[0]), np.max(RGI_oi[0]), 
        np.min(RGI_oi[1]), np.max(RGI_oi[1]))

buffer_pix = np.round(buffer_bbox/geoTransform[1])

bbox = (np.amax(( 0, bbox[0]-buffer_pix)), 
        np.amin(( RGI.shape[1], bbox[1]+buffer_pix)),
        np.amax(( 0, bbox[2]-buffer_pix)),
        np.amin(( RGI.shape[0], bbox[3]+buffer_pix))
        )
