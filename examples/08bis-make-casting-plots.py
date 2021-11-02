import os
import getpass

import numpy as np

import pystac
import stac2dcache
from stac2dcache.utils import copy_asset, get_asset, catalog2geopandas 

import rioxarray

from skimage.color import hsv2rgb
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from eratosthenes.generic.handler_dat import get_list_files
from eratosthenes.generic.mapping_tools import pix2map, map2pix
from eratosthenes.generic.mapping_io import read_geo_image
from eratosthenes.postprocessing.solar_tools import make_shading



def read_catalog(url):
    """
    Read STAC catalog from URL
    
    :param url: urlpath to the catalog root
    :return: PySTAC Catalog object
    """
    url = url if url.endswith("catalog.json") else f"{url}/catalog.json"
    catalog = pystac.Catalog.from_file(url)
    return catalog

dcache_root = (
    "https://webdav.grid.surfsara.nl:2880"
    "/pnfs/grid.sara.nl/data/eratosthenes/"
    "disk/"
    )
dem_index_url = dcache_root + "ArcticDEM_tiles_sentinel-2"
rgi_index_url = dcache_root + "RasterRGI_tiles_sentinel-2"
# catalog with shadow-enhanced images
catalog_id = "red-glacier_sentinel-2_shadows"
catalog_url = dcache_root + catalog_id

tile_id = '5VMG'
rgi_id = 'RGI60-01.19773' # Red Glacier
bbox = (4353, 5279, 9427, 10980) # 1000 m buffer

# configure connection to dCache
username = getpass.getuser()
if username=='Alten005':
    dcache = stac2dcache.configure(
        filesystem="dcache", 
        token_filename=os.path.expanduser("~/Eratosthenes/HPC/macaroon.dat"),
        )
else:
    dcache = stac2dcache.configure(
        filesystem="dcache", 
        token_filename="macaroon.dat"
        )
dcache.webdav_url="https://webdav.grid.surfsara.nl:2880"

if not os.path.exists(tile_id+'_arcDEM.tif'):
    # download assets - from web to storage
    dcache.download(os.path.join(dem_index_url, tile_id+'.tif'), 
                    tile_id+'_arcDEM.tif')
if not os.path.exists(tile_id+'_RGI.tif'):    
    dcache.download(rgi_index_url+'/'+tile_id+'.tif', 
                    tile_id+'_RGI.tif')

# read DEM index
Z, spatialRef, geoTransform, targetprj = read_geo_image(tile_id+'_arcDEM.tif')
R, spatialRef, geoTransform, targetprj = read_geo_image(tile_id+'_RGI.tif')

# get subset
Z = Z[bbox[0]:bbox[1],bbox[2]:bbox[3]]
R = R[bbox[0]:bbox[1],bbox[2]:bbox[3]]
R = ( R== int(rgi_id.split('.')[1]) )

x_new, y_new = pix2map(geoTransform, bbox[0], bbox[2])
geoTransform = (
    x_new, geoTransform[1], geoTransform[2], 
    y_new, geoTransform[4], geoTransform[5]
    )
# make shading image
S = make_shading(Z, 45, 45, spac=10)

# make composite

Red_rgb = hsv2rgb( np.dstack((R.astype(float)*.55, 
                              R.astype(float)*.7, # .5*np.ones(R.shape) 
                              (S*.2)+.8 )) )
Red_rgb = np.uint8(Red_rgb * 2**8)

img = Image.fromarray(Red_rgb)
img.save("Red-bg.jpg", quality=95)

# find casting data

# read image catalog
catalog = read_catalog(catalog_url)
# get all items as a generator
items = catalog.get_all_items()

# convert catalog to geopandas dataframe 
#gdf = catalog2geopandas(catalog, crs='EPSG:4326')
# identify which scenes have a specific properties 
# Example: cloud cover lower than 50%
#mask = gdf["eo:cloud_cover"] < 50.
#subset = gdf[mask]
#item_ids = subset.index

if not os.path.exists('conn'):
    os.mkdir('conn')
    
for item in items:
    if item.properties['eo:cloud_cover']<=50:        
        item_url = item.assets['connectivity'].href
        item_date = item.datetime
        
        dcache.download(item_url, 
                        os.path.join( 'conn',
                                      'conn-' + item_date.strftime("%Y-%m-%d") 
                                      + '.txt') 
                        )
# for item in items:
#     #print(item.datetime.strftime("%Y-%m-%d"))
#     if item.datetime.strftime("%Y-%m-%d")=='2020-04-05':        
#         item_url = item.assets['shadow'].href
#         item_date = item.datetime
#         print('found!')
#         dcache.download(item_url, 
#                         os.path.join( 'conn',
#                                      'shdw-' + item_date.strftime("%Y-%m-%d") 
#                                      + '.tif') 
#                         )




# loop through textfiles

txt_list = get_list_files(os.path.join(os.getcwd(), 'conn'), '.txt')

# make selection of year 2020

for txt_file in txt_list:
    cast_list = np.loadtxt(os.path.join('conn', txt_file))
    # transform to pixel coordinates
    shdw_i, shdw_j = map2pix(geoTransform, cast_list[:,2], cast_list[:,3])
    
    # make selection if shadow is on glacier
    shdw_i_int, shdw_j_int = np.round(shdw_i).astype(int), np.round(shdw_j).astype(int)    
    IN = (shdw_i_int>=0) & (shdw_i_int<R.shape[0]) & \
        (shdw_j_int>=0) & (shdw_j_int<R.shape[1])        
    shdw_i_int, shdw_j_int = shdw_i_int[IN], shdw_j_int[IN]
    
    IN_glac = R[shdw_i_int, shdw_j_int]
    
    cast_list = cast_list[IN,:]
    cast_list = cast_list[IN_glac,:]
    
    # transform to pixel coordinates
    ridg_i, ridg_j = map2pix(geoTransform, cast_list[:,0], cast_list[:,1])
    shdw_i, shdw_j = map2pix(geoTransform, cast_list[:,2], cast_list[:,3])
    
    #im = Image.open("Red-test.png")
    im = Image.new("RGBA", (1553, 926))
    
    draw = ImageDraw.Draw(im)
    for i in np.arange(0,len(ridg_i),4):
        draw.line((ridg_j[i], ridg_i[i], 
                   shdw_j[i], shdw_i[i]), 
                  fill=(255,0,64,255), 
                  width=1)
    # write to stdout
    im.save(os.path.join('conn', txt_file[:-4]+".png"))


