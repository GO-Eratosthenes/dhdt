# compare with ICESat-2, from openaltimetry.org
# with elevation from CopDEM
# generate hypsometric curve


# generic libraries
import numpy as np

# import geospatial libraries
import osr, ogr, gdal

# inhouse functions
from eratosthenes.generic.handler_dat import get_list_files

glac_name = 'Red Glacier'
rgi_id = 'RGI60-01.19773'
# source directories
src_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/ICESat2/'
rgi_path = '/Users/Alten005/surfdrive/Eratosthenes/Denali/GIS/'
rgi_name = '01_rgi60_alaska.shp'

## processing directory
prc_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Processing/'

csv_list = get_list_files(src_path, '.csv')

for i in range(len(csv_list)):    
    file_path = src_path + csv_list[i]
    dat_time = np.datetime64(csv_list[i][5:15])
    xyz = np.loadtxt(file_path, skiprows=1, usecols=(1,2,3), \
                     delimiter=",") # lon, lat, elev
    xyz = np.pad(xyz, ((0,0), (0,1)), 'constant', constant_values=(dat_time))   
    if i ==0:
        xyzt = xyz
    else:       
        xyzt = np.concatenate((xyzt, xyz), axis=0)
    

## get glacier outline
drv = ogr.GetDriverByName("ESRI Shapefile")
shp = drv.Open(rgi_path+rgi_name)
lyr = shp.GetLayer()

feat = lyr.GetNextFeature()
while feat:     
        if feat.GetField('RGIId')==rgi_id:    
            geom = feat.GetGeometryRef() # get the glacier geometry      
        feat = lyr.GetNextFeature() 
lyr.ResetReading()

# exclude off glacier ICESat points

pt = ogr.Geometry(ogr.wkbPoint)
pt.AssignSpatialReference(geom.GetSpatialReference)
pt.SetPoint(0, xyzt[0,0], xyzt[0,1])


# get elevation        

