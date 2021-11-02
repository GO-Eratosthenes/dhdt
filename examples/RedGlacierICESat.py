# compare with ICESat-2, from openaltimetry.org
# with elevation from CopDEM
# generate hypsometric curve


# generic libraries
import numpy as np

# import geospatial libraries
from osgeo import osr, ogr, gdal

# inhouse functions
from eratosthenes.generic.handler_dat import get_list_files
from eratosthenes.generic.mapping_io import read_geo_image
from eratosthenes.generic.mapping_tools import bilinear_interp_excluding_nodat, map2pix 

glac_name = 'Red Glacier'
rgi_id = 'RGI60-01.19773'
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
    
# get elevation  
dem_list = get_list_files(dem_path, '.tif')

cop_dem = np.zeros((len(xyzt)), dtype=np.float)
for i in range(len(dem_list)):      
    DEM,_,geoTran_DEM,_ = read_geo_image( dem_path + dem_list[i])
    i_DEM, j_DEM = map2pix(geoTran_DEM, xyzt[:,0], xyzt[:,1])
    
    in_DEM = (i_DEM>0) & (i_DEM<DEM.shape[0]-1) & (j_DEM>0) & (j_DEM<DEM.shape[1]-1)
    
    #in_dem = 
    z = bilinear_interp_excluding_nodat(DEM, i_DEM[in_DEM], j_DEM[in_DEM], \
                                        -9999)
    cop_dem[in_DEM] = z
    del in_DEM, i_DEM, j_DEM, z
    
# plot elevation over time....
dz = cop_dem - xyzt[:,2]

# temporal grouping
yr_beg = np.datetime64('1970-01-01','D') # unix time
yr_min = yr_beg + np.timedelta64(np.amin(xyzt[:,3]).astype(int), 'D')
yr_min = yr_min.astype('datetime64[Y]').astype(int) + 1970
yr_max = yr_beg + np.timedelta64(np.amax(xyzt[:,3]).astype(int), 'D')
yr_max = yr_max.astype('datetime64[Y]').astype(int) + 1970
yr_range = np.arange(yr_min, yr_max+1)

z_spac = 100 # bin spacing
z_bins = np.arange(np.floor(np.amin(xyzt[:,2])/z_spac)*z_spac +.5*z_spac, \
                   np.ceil(np.amax(xyzt[:,2])/z_spac)*z_spac -.5*z_spac, \
                   z_spac)

z_vs_dz = np.zeros((yr_max-yr_min+1, len(z_bins), 3), dtype=float)    
for yr_idx in range(len(yr_range)):
    yr = yr_range[yr_idx]
    in_dat = (xyzt[:,3] > np.datetime64(str(yr)+'-01-01').astype(int)) & \
        (xyzt[:,3] < np.datetime64(str(yr+1)+'-01-01').astype(int))
    z_idx = np.digitize(xyzt[in_dat,2], z_bins)
    dz_yr = dz[in_dat]
    z_idx_yr = np.unique(z_idx)
    for idx in z_idx_yr:
        z_vs_dz[yr_idx, idx-1, 0] = np.median(dz_yr[z_idx==idx])
        z_vs_dz[yr_idx, idx-1, 1] = np.quantile(dz_yr[z_idx==idx], 0.25)
        z_vs_dz[yr_idx, idx-1, 2] = np.quantile(dz_yr[z_idx==idx], 0.75)
    
import matplotlib.pyplot as plt
    
#plt.bar(z_bins, z_vs_dz[0,:,0], width=z_spac/2, yerr=z_vs_dz[0,:,1:3].T)
plt.bar(z_bins, z_vs_dz[1,:,0], width=z_spac/2, yerr=z_vs_dz[1,:,1:3].T)
#plt.bar(z_bins, z_vs_dz[2,:,0], width=z_spac/2, yerr=z_vs_dz[2,:,1:3].T)
plt.show()




