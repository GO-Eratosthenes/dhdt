# generic libraries
import numpy as np

# import geospatial libraries
from osgeo import osr, ogr, gdal
from pyproj import Transformer

from skimage.color import hsv2rgb
from PIL import Image, ImageDraw

# inhouse functions
from eratosthenes.generic.handler_dat import get_list_files
from eratosthenes.generic.mapping_io import read_geo_image
from eratosthenes.generic.mapping_tools import pix2map, map2pix
from eratosthenes.generic.mapping_tools import bilinear_interp_excluding_nodat 

glac_name = 'Red Glacier'
rgi_id = 'RGI60-01.19773'
tile_id = '5VMG'
bbox = (4353, 5279, 9427, 10980) # 1000 m buffer

# source directories
src_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/ICESat2/'
rgi_path = '/Users/Alten005/surfdrive/Eratosthenes/Denali/GIS/'
rgi_name = '01_rgi60_alaska.shp'
dem_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Cop-DEM-GLO-30/'

## processing directory
prc_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Processing/'

csv_list = get_list_files(src_path, '.csv')

for i in range(len(csv_list)):    
    file_path = src_path + csv_list[i]
    dat_time = np.datetime64(csv_list[i][5:15],'D') # days since 1970-01-01
    xyz = np.loadtxt(file_path, skiprows=1, usecols=(1,2,3), \
                     delimiter=",") # lon, lat, elev
    xyz = np.pad(xyz, ((0,0), (0,1)), 'constant', constant_values=(dat_time))   
    if i ==0:
        xyzt = xyz
    else:       
        xyzt = np.concatenate((xyzt, xyz), axis=0)
    del xyz, dat_time

## get glacier outline
drv = ogr.GetDriverByName("ESRI Shapefile")
shp = drv.Open(rgi_path+rgi_name)
lyr = shp.GetLayer()

feat = lyr.GetNextFeature()
while feat:     
        if feat.GetField('RGIId')==rgi_id:    
            #geom = feat.GetGeometryRef() # get the glacier geometry  
            geom = feat.geometry().Clone() # get the glacier geometry  
        feat = lyr.GetNextFeature() 
lyr.ResetReading()

# exclude off glacier ICESat points
in_glac = np.zeros((len(xyzt)), dtype=bool)
for i in range(len(xyzt)):
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(xyzt[i,0], xyzt[i,1])
    if geom.Contains(pt):
        in_glac[i] = True
xyzt = xyzt[in_glac,:]
del pt, geom, in_glac

# get local reference frame
I, spatialRef, geoTransform, targetprj = \
    read_geo_image('T05VMG_20200810T213541_B04.jp2')
I = I[bbox[0]:bbox[1],bbox[2]:bbox[3]]

x_new, y_new = pix2map(geoTransform, bbox[0], bbox[2])
geoTransform = (
    x_new, geoTransform[1], geoTransform[2], 
    y_new, geoTransform[4], geoTransform[5]
    )

transformer = Transformer.from_crs("epsg:4326", "epsg:32605")
x_is,y_is = transformer.transform(xyzt[:,1], xyzt[:,0])
i_is, j_is = map2pix(geoTransform, x_is, y_is)

i_is = np.round(i_is).astype(int)
j_is = np.round(j_is).astype(int)

im = Image.open("Red-bg.png")
I = np.array(im)

for di in np.array([-1,0,+1]):
    for dj in np.array([-1,0,+1]):
        I[i_is+di, j_is+dj, 0] = 255
        I[i_is+di, j_is+dj, 1] = 0
        I[i_is+di, j_is+dj, 2] = 128

im = Image.fromarray(I)
im.save("Red-IceSat.jpg", quality=95)

