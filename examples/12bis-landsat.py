import os
import numpy as np

from osgeo import ogr, osr, gdal

from eratosthenes.generic.handler_ls import meta_LSstring, get_LS_footprint
from eratosthenes.generic.mapping_io import read_geo_image
from eratosthenes.generic.mapping_tools import pix2map
from eratosthenes.processing.coupling_tools import match_pair
from eratosthenes.processing.matching_tools import get_coordinates_of_template_centers

fpath1 = os.path.join('/Users/Alten005/Eratosthenes/assessment/data/L7_StElias', \
                      'LE07_L1TP_062018_20190829_20190924_01_T1_B8.TIF')
fpath2 = os.path.join('/Users/Alten005/Eratosthenes/assessment/data/L7_StElias', \
                      'LC08_L1TP_062018_20190906_20190917_01_T1_B8.TIF')
wrs_path = os.path.join('/Users/Alten005/surfdrive/Eratosthenes/Denali/GIS/', \
                        'wrs2_descending.shp')
dat_dir = '/Users/Alten005/GO-eratosthenes/start-code/examples/'

L7_dir, L7_name = os.path.split(fpath1)
(L7_t, L7_pth, L7_row) = meta_LSstring(L7_name)

# get imagery data
(L7, spatialRefL7, geoTransformL7, targetprjL7) = read_geo_image(fpath1)
(L8, spatialRefL8, geoTransformL8, targetprjL8) = read_geo_image(fpath2)
(L7_Msk,_,_,_) = read_geo_image(os.path.join(L7_dir, 'LE07_L1TP_062018_20190829_20190924_01_T1_GM_B8.TIF'))
L8_Msk = L8!=0

# get footprint in lat-lon
L7_footp, wrsSpatRef = get_LS_footprint(wrs_path, L7_pth, L7_row)

# do coordinate transformation from sperical lat-lon to metric projection
L7SpatRef = osr.SpatialReference()
L7SpatRef.ImportFromWkt(spatialRefL7)
coordTrans = osr.CoordinateTransformation(wrsSpatRef, L7SpatRef)

# loop seems to be needed....
L7_footp_xy = np.zeros(L7_footp.shape, dtype=float)
L7_footp_geom = ogr.Geometry(ogr.wkbLinearRing)
for p in range(L7_footp.shape[0]):
    x,y,_ = coordTrans.TransformPoint(L7_footp[p,0],L7_footp[p,1])
    L7_footp_xy[p,0], L7_footp_xy[p,1] = x, y
    L7_footp_geom.AddPoint_2D(x, y)
L7_footp_poly = ogr.Geometry(ogr.wkbPolygon)
L7_footp_poly.AddGeometry(L7_footp_geom)

grd_step = 32
temp_radius, search_radius = 32, 32
# create grid
Igrd, Jgrd = get_coordinates_of_template_centers(L7, grd_step)
# generate grid in X,Y
Xgrd, Ygrd = pix2map(geoTransformL7, Igrd, Jgrd)

# find gridpoints within footprint
IN = np.zeros(Xgrd.shape, dtype=bool)
for p in range(Xgrd.size):
    lin_idx = np.unravel_index(p, Xgrd.shape)
    wkt_str = 'POINT (' + str(Xgrd[lin_idx]) + ' ' + str(Ygrd[lin_idx]) + ')' 
    grd_point = ogr.CreateGeometryFromWkt(wkt_str)
    if L7_footp_poly.Contains(grd_point):
        IN[lin_idx] = True

Xgrd, Ygrd = Xgrd[IN], Ygrd[IN]


match_pair(L7, L8, L7_Msk, L8_Msk, geoTransformL7, geoTransformL8, \
               temp_radius, search_radius, Xgrd[1234], Ygrd[1234], \
               kernel='kroon', \
               prepro='bandpass', match='cosicorr', subpix='tpss', \
               processing='stacking', boi=np.array([]) )



(L7_footp_x, L7_footp_y,_) = coordTrans.TransformPoints(L7_footp[:,0],L7_footp[:,1])
L7_footp.Transform(coordTrans)            
   
 




