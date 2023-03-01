import os

import numpy as np

from dhdt.auxilary.handler_copernicusdem import \
    get_copDEM_s2_tile_intersect_list, download_and_mosaic_through_ftps, \
    get_cds_path_copDEM, get_cds_url
from dhdt.generic.mapping_tools import get_bbox
from dhdt.generic.mapping_io import make_im_from_geojson
from dhdt.generic.handler_sentinel2 import get_generic_s2_raster
from dhdt.auxilary.handler_randolph import create_rgi_tile_s2
from dhdt.auxilary.handler_coastal import get_coastal_dataset

#import geopandas
#geom_dir = '/Users/Alten005/surfdrive/Eratosthenes/Coastline'
#geom_name = 'GSHHS_f_L1.shp'
#geom_path = get_coastal_dataset(geom_dir, minimal_level=1, resolution='f')

#geom_coast = geopandas.read_file(geom_path)

#toi = '15MXV'
#from dhdt.auxilary.handler_coastal import get_coastal_polygon_of_tile
#aap = get_coastal_polygon_of_tile(toi)
#geoTransform, crs = get_generic_s2_raster(toi)
#make_im_from_geojson(geoTransform,crs,'test.tif', 'GSHHS_f_L1.geojson')


sso = 'basalt'
pw = '!Q2w3e4r5t6y7u8i9o0p'
toi = '09VWJ'

dem_path = '/Users/Alten005/surfdrive/Eratosthenes/Bologna/Cop-DEM_GLO-30/'
im_path = '/Users/Alten005/surfdrive/Eratosthenes/Bologna/S2/'
rgi_path = '/Users/Alten005/surfdrive/Eratosthenes/Bologna/GIS/'

dem_name = 'COP-DEM-'+toi+'.tif'
rgi_name = 'test.tif'

urls,tars = get_copDEM_s2_tile_intersect_list(toi)

geoTransform, crs = get_generic_s2_raster(toi)

if not os.path.exists(os.path.join(dem_path,dem_name)):
    # create CopernicusDEM grid for MGRS tile
    bbox = get_bbox(geoTransform)
    bbox = bbox.reshape((2, 2)).T.ravel()  # x_min, y_min, x_max, y_max

    cds_path, cds_url = get_cds_path_copDEM(), get_cds_url()
    cop_dem = download_and_mosaic_through_ftps(tars, dem_path,
                                               cds_url, cds_path[4:],
                                               sso, pw, bbox, crs, geoTransform)

    cop_dem.rio.to_raster(os.path.join(dem_path,dem_name))

if not os.path.exists(os.path.join(rgi_path,rgi_name)):
    # create glacier inventory grid for MGRS tile
    create_rgi_tile_s2(toi, geom_out=rgi_path)
    print('.')