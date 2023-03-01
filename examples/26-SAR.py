import os
import numpy as np

from skimage import data

from dhdt.auxilary.handler_randolph import create_rgi_tile_s2
from dhdt.generic.handler_cop import make_copDEM_mgrs_tile
from dhdt.generic.mapping_io import read_geo_image
from dhdt.postprocessing.solar_tools import make_doppler_range
from dhdt.postprocessing.group_statistics import get_general_hypsometry

dem_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Cop-DEM_GLO-30/'
bal_dir = '/Users/Alten005/Quaternion/RS2baltoro'

sso = 'basalt'
pw = '!Q2w3e4r5t6y7u8i9o0p'
poi = np.array([46.43136, 11.82774])
toi = '32TQS'
#poi = np.array([35.73446, 76.46439])
#toi = '43SFV'

# get RGI
dat_path = '/Users/Alten005/surfdrive/Eratosthenes/Denali/'
rgi_path = dat_path + 'GIS/'


rgi_name = create_rgi_tile_s2(rgi_path, poi, toi, rgi_out=None)
rgi_name = rgi_name.split('/')[-1]

sso = 'basalt'
pw = '!Q2w3e4r5t6y7u8i9o0p'
dem_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Cop-DEM_GLO-30/'
s2_path = '/Users/Alten005/surfdrive/Eratosthenes/SatelliteTiles'

cop_dem_mgrs = os.path.join(bal_dir, 'COP-DEM-'+toi+'.tif')
if not os.path.exists(cop_dem_mgrs):
    cop_dem_mgrs = make_copDEM_mgrs_tile(toi, rgi_path, rgi_name, s2_path, dem_path,
         tile_name='sentinel2_tiles_world.shp', cop_name='mapping.csv',
         map_name='DGED-30.geojson', sso=sso, pw=pw, out_path=bal_dir)

Z = read_geo_image(cop_dem_mgrs)[0]

a,b = get_general_hypsometry(Z, np.random.random(Z.shape))

S = make_doppler_range(Z, 30, 60)
