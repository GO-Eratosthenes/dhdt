import pathlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

from eratosthenes.generic.handler_s2 import meta_S2string
from eratosthenes.generic.mapping_io import makeGeoIm, read_geo_image, \
    read_geo_info
from eratosthenes.generic.gis_tools import ll2utm, shape2raster
from eratosthenes.generic.handler_im import get_image_subset

from eratosthenes.preprocessing.handler_multispec import create_shadow_image, \
    create_caster_casted_list_from_polygons
from eratosthenes.preprocessing.shadow_geometry import create_shadow_polygons

from eratosthenes.processing.coregistration import coregister, \
    get_coregistration
from eratosthenes.processing.network_tools import getNetworkBySunangles

dat_path = '/Users/Alten005/surfdrive/Eratosthenes/Denali/'
dat_path = pathlib.Path(dat_path)

tile = 'T05VNK'
scene_paths = [path for path in dat_path.rglob(f'S2?_MSIL1C_*_{tile}_*')]

shadow_transform = 'ruffenacht'  # method to deploy for shadow enhancement

# do a subset of the imagery
minI = 4000  # minimal row coordinate
maxI = 6000  # maximum row coordinate
minJ = 4000  # minimal column coordinate
maxJ = 6000  # maximum column coordinate

bbox = (minI, maxI, minJ, maxJ)

for scene_path in scene_paths:
    # (sat_time,sat_orbit,sat_tile) = meta_S2string(scene_path.name)
    print(f'working on {scene_path.name}')
    shadow_path = scene_path / 'shadows.tif'
    if not shadow_path.exists():
        M, geoTransform, crs = create_shadow_image(scene_path,
                                                   shadow_transform,
                                                   bbox)
        print(f'produced shadow transform for {scene_path.name}')
        makeGeoIm(M, geoTransform, crs, shadow_path)
    else:
        M, crs, geoTransform, targetprj = read_geo_image(shadow_path)

    labels_path = scene_path / 'labelPolygons.tif'
    cast_conn_path = scene_path / 'labelCastConn.tif'
    if not (labels_path.exists() and cast_conn_path.exists()):
        labels, cast_conn = create_shadow_polygons(M, scene_path, bbox)
        makeGeoIm(labels, geoTransform, crs, labels_path)
        makeGeoIm(cast_conn, geoTransform, crs, cast_conn_path)
        print(f'labelled shadow polygons for {scene_path.name}')

# make raster with Randolph glacier mask for the tile
rgi_shape_path = dat_path / 'GIS/01_rgi60_Alaska.shp'
rgi_raster_path = (dat_path / tile).with_suffix('.tif')
if not rgi_raster_path.exists():
    # create RGI raster for the extent of the image
    rgi_shape_utm_path = rgi_shape_path.with_name(rgi_shape_path.stem
                                                  + f'_utm{tile[1:3]}')
    rgi_shape_utm_path = rgi_shape_utm_path.with_suffix(rgi_shape_path.suffix)
    # get geo-meta data for a tile
    fname = [im for im in scene_paths[0].rglob('*B02.jp2')][0]
    crs, geoTransform, targetprj, rows, cols, bands = read_geo_info(fname)
    aoi = 'RGIId'
    if not rgi_shape_utm_path.exists():  # project RGI shapefile
        # transform shapefile from lat-long to UTM
        ll2utm(rgi_shape_path, rgi_shape_utm_path, crs, aoi)
    # convert polygon file to raster file
    shape2raster(rgi_shape_utm_path, rgi_raster_path, geoTransform,
                 rows, cols, aoi)

rgi_mask, crs, geoTransform, targetprj = read_geo_image(rgi_raster_path)
rgi_mask = get_image_subset(rgi_mask, bbox)

# make stack of Labels & Connectivity
for scene_path in scene_paths:
    if not (scene_path / 'conn.txt').exists():
        create_caster_casted_list_from_polygons(scene_path,
                                                rgi_raster_path,
                                                bbox)

connectivity = 2
coreg_path = dat_path / 'coreg.txt'
if not coreg_path.exists():
    coregister(scene_paths, coreg_path, connectivity=connectivity,
               step_size=True, temp_size=15, bbox=bbox, lstsq_mode='ordinary')
# lkTransform = RefScale(RefTrans(subTransform,tempSize/2,tempSize/2),tempSize)
# makeGeoIm(Dstack[0],lkTransform,crs,"DispAx1.tif")

# get co-registration information
connectivity_pairs = getNetworkBySunangles(scene_paths, connectivity)
coName, coReg = get_coregistration(coreg_path)

# construct connectivity
for connectivity_pair in connectivity_pairs:
    scene_path1 = scene_paths[connectivity_pair[0]]
    scene_path2 = scene_paths[connectivity_pair[1]]

    # get start and finish points of shadow edges
    conn1 = np.loadtxt(fname=scene_path1 / 'conn.txt')
    conn2 = np.loadtxt(fname=scene_path2 / 'conn.txt')

    # compensate for coregistration
    coid1 = coName.index(scene_path1.as_posix())
    coid2 = coName.index(scene_path2.as_posix())
    coDxy = coReg[coid1] - coReg[coid2]
    conn2[:, 0] += coDxy[0]
    conn2[:, 1] += coDxy[1]
    # conn2[:,2] += coDxy[0]
    # conn2[:,3] += coDxy[1]

    # find nearest
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(conn1[:, 0:2])
    distances, indices = nbrs.kneighbors(conn2[:, 0:2])
    IN = distances < 20
    idxConn = np.transpose(np.vstack((np.where(IN)[0],
                                      indices[distances < 20])))

# walk through glacier polygons
# rgiList = np.trim_zeros(np.unique(Rgi))
# for j in rgiList:
# selection of glacier polygon
# selec = Rgi==j

# keep it image-based
selec = rgi_mask != 0

# linIdx = np.where(IN)
# listOfCoordinates= list(zip(linIdx[0], linIdx[1], linIdx[2]))

# find associated caster
fig, ax = plt.subplots()
im = ax.imshow(selec.astype(int))
fig.colorbar(im)
plt.show()

# processed_image = cv2.filter2D(image,-1,kernel)

fig, ax = plt.subplots()
im = ax.imshow(rgi_mask)
fig.colorbar(im)
plt.show()
