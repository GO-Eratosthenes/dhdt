import os
import geopandas
import ftps

import rioxarray

from dhdt.generic.handler_www import get_file_from_ftps
from dhdt.auxilary.handler_cop import \
    get_copDEM_s2_tile_intersect_list, download_and_mosaic_through_ftps, \
    get_cds_path_copDEM, get_cds_url
from dhdt.generic.mapping_io import read_geo_info
from dhdt.generic.mapping_tools import get_bbox
from dhdt.generic.handler_sentinel2 import get_crs_from_mgrs_tile

sso = 'basalt'
pw = '!Q2w3e4r5t6y7u8i9o0p'
toi = '05VMG'

dem_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Cop-DEM_GLO-30/'
cop_file = 'DGED-30.geojson'
s2_path = '/Users/Alten005/surfdrive/Eratosthenes/SatelliteTiles'
s2_name = 'sentinel2_tiles_world.shp'
rgi_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Cop-DEM-GLO-30/'
rgi_file = '5VMG_RGI.tif' # '5VMG_RGI_red.tif'

urls,tars = get_copDEM_s2_tile_intersect_list(s2_path, dem_path, toi,
      s2_file='sentinel2_tiles_world.shp', cop_file='mapping.csv',
      map_file='DGED-30.geojson')

crs = get_crs_from_mgrs_tile(toi)
_,geoTransform,_,rows,cols,_ = read_geo_info(os.path.join(rgi_path, rgi_file))
bbox = get_bbox(geoTransform, rows, cols)
bbox = bbox.reshape((2,2)).T.ravel() # x_min, y_min, x_max, y_max

cds_path, cds_url = get_cds_path_copDEM(), get_cds_url()
cop_dem = download_and_mosaic_through_ftps(tars, dem_path,
                                           cds_url, cds_path[4:],
                                           sso, pw, bbox, crs, geoTransform)

cop_dem.rio.to_raster(os.path.join(dem_path,'COP-DEM-'+toi+'.tif'))

cds_url = 'cdsdata.copernicus.eu:990'
dem_url =urls[0]
tar_name = tars[0]

get_file_from_ftps(cds_url, sso, pw, cds_path[4:], tar_name, dem_path)

client = ftps.FTPS('ftps://' + sso+':' +pw+ '@cdsdata.copernicus.eu:990')
client.list()
client.download(cds_path + tar_name, "tmp.tar")

