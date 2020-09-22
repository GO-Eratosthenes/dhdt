import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

from eratosthenes.generic.handler_s2 import meta_S2string
from eratosthenes.generic.mapping_tools import RefTrans, pix2map, map2pix, \
    castOrientation
from eratosthenes.generic.mapping_io import makeGeoIm, read_geo_image, \
    read_geo_info
from eratosthenes.generic.gis_tools import ll2utm, shape2raster

from eratosthenes.preprocessing.handler_multispec import create_shadow_image
from eratosthenes.preprocessing.read_s2 import read_sun_angles_s2
from eratosthenes.preprocessing.shadow_geometry import create_shadow_polygons

from eratosthenes.processing.handler_s2 import read_view_angles_s2
from eratosthenes.processing.network_tools import getNetworkIndices, \
    getNetworkBySunangles
from eratosthenes.processing.matching_tools import LucasKanade

dat_path = '/Users/Alten005/surfdrive/Eratosthenes/Denali/' 
#im_path = 'Data/S2A_MSIL1C_20180225T214531_N0206_R129_T05VPL_20180225T232042/'
#fName = 'T05VPL_20180225T214531_B'

im_path = ('Data/S2A_MSIL1C_20180225T214531_N0206_R129_T05VNK_20180225T232042/',
          'Data/S2B_MSIL1C_20190225T214529_N0207_R129_T05VNK_20190226T010218/',
          'Data/S2A_MSIL1C_20200225T214531_N0209_R129_T05VNK_20200225T231600/')
fName = ('T05VNK_20180225T214531_B', 'T05VNK_20190225T214529_B', 'T05VNK_20200225T214531_B')

shadow_transform = 'ruffenacht' # method to deploy for shadow enhancement

# do a subset of the imagery
minI = 4000 # minimal row coordiante
maxI = 6000 # maximum row coordiante
minJ = 4000 # minimal collumn coordiante
maxJ = 6000 # maximum collumn coordiante

for i in range(len(im_path)):
    sen2Path = dat_path + im_path[i]
    (sat_time,sat_orbit,sat_tile) = meta_S2string(im_path[i])
    print('working on '+ fName[i][0:-2])
    if not os.path.exists(sen2Path + 'shadows.tif'):
        (M, geoTransform, crs) = create_shadow_image(dat_path, im_path[i], \
                                                   shadow_transform, \
                                                   minI, maxI, minJ, maxJ \
                                                   )
        print('produced shadow transform for '+ fName[i][0:-2])
        makeGeoIm(M, geoTransform, crs, sen2Path + 'shadows.tif')
    else:
        (M, crs, geoTransform, targetprj) = read_geo_image(
            sen2Path + 'shadows.tif')

    if not os.path.exists(sen2Path + 'labelCastConn.tif'):
        (labels, cast_conn) = create_shadow_polygons(M,sen2Path, \
                                                     minI, maxI, minJ, maxJ \
                                                     )

        makeGeoIm(labels, geoTransform, crs, sen2Path + 'labelPolygons.tif')
        makeGeoIm(cast_conn, geoTransform, crs, sen2Path + 'labelCastConn.tif')
        print('labelled shadow polygons for '+ fName[i][0:-2])

# make raster with Randolph glacier mask for the tile
if not os.path.exists(dat_path + sat_tile + '.tif'):
    # create RGI raster for the extent of the image
    rgi_path = dat_path+'GIS/'
    rgi_file = '01_rgi60_Alaska.shp'
    out_shp = rgi_path+rgi_file[:-4]+'_utm'+sat_tile[1:3]+'.shp'
    # get geo-meta data for a tile
    fname = dat_path + im_path[0] + fName[0] + '04.jp2'
    crs, geoTransform, targetprj, rows, cols, bands = read_geo_info(fname)
    aoi = 'RGIId'
    if not os.path.exists(out_shp):  # project RGI shapefile
        # transform shapefile from lat-long to UTM
        ll2utm(rgi_path+rgi_file,out_shp,crs,aoi)
    # convert polygon file to raster file
    shape2raster(out_shp, dat_path+sat_tile, geoTransform, rows, cols, aoi)

(rgi_mask, crs, geoTransform, targetprj) = read_geo_image(dat_path
                                                          +sat_tile+'.tif')
rgi_mask = rgi_mask[minI:maxI,minJ:maxJ]

## processing

## co-register

# construct network
GridIdxs = getNetworkIndices(len(im_path))
connectivity = 2 # amount of connection
GridIdxs = getNetworkBySunangles(dat_path, im_path, connectivity)

Astack = np.zeros([GridIdxs.shape[1],len(im_path)]) # General coregistration adjustment matrix
Astack[np.arange(GridIdxs.shape[1]),GridIdxs[0,:]] = +1
Astack[np.arange(GridIdxs.shape[1]),GridIdxs[1,:]] = -1
Astack = np.transpose(Astack)

tempSize = 15
stepSize = True
# get observation angle
(obsZn,obsAz) = read_view_angles_s2(sen2Path)
obsZn = obsZn[minI:maxI,minJ:maxJ]
obsAz = obsAz[minI:maxI,minJ:maxJ]
if stepSize: # reduce to kernel resolution
    radius = np.floor(tempSize/2).astype('int')
    Iidx = np.arange(radius, obsZn.shape[0]-radius, tempSize)
    Jidx = np.arange(radius, obsZn.shape[1]-radius, tempSize)
    IidxNew = np.repeat(np.transpose([Iidx]), len(Jidx), axis=1)
    Jidx = np.repeat([Jidx], len(Iidx), axis=0)
    Iidx = IidxNew

    obsZn = obsZn[Iidx,Jidx]
    obsAz = obsAz[Iidx,Jidx]
    del IidxNew,radius,Iidx,Jidx

sig_y = 10; # matching precision
Dstack = np.zeros([GridIdxs.shape[1],2])
for i in range(GridIdxs.shape[1]):
    for j in range(2):
        sen2Path = dat_path + im_path[GridIdxs[j,i]]
        # get sun orientation
        (sunZn,sunAz) = read_sun_angles_s2(sen2Path)
        sunZn = sunZn[minI:maxI,minJ:maxJ]
        sunAz = sunAz[minI:maxI,minJ:maxJ]
        # read shadow image
        img = gdal.Open(sen2Path + "shadows.tif")
        M = np.array(img.GetRasterBand(1).ReadAsArray())
        # create ridge image
        Mcan = castOrientation(M,sunZn,sunAz)
        # Mcan = castOrientation(msk,sunZn,sunAz)
        Mcan[Mcan<0] = 0
        if j==0:
            Mstack = Mcan
        else:
            Mstack = np.stack((Mstack, Mcan), axis=2)

    # optical flow following Lukas Kanade

    #####################
    # blur with gaussian?
    #####################

    (y_N,y_E) = LucasKanade(Mstack[:,:,0], Mstack[:,:,1],
                    tempSize, stepSize)
    # select correct displacements
    Msk = y_N!=0

    #########################
    # also get stable ground?
    #########################

    # robust selection through group statistics
    med_N = np.median(y_N[Msk])
    mad_N = np.median(np.abs(y_N[Msk] - med_N))

    med_E = np.median(y_E[Msk])
    mad_E = np.median(np.abs(y_E[Msk] - med_E))

    # robust 3sigma-filtering
    alpha = 3
    IN_N = (y_N > med_N-alpha*mad_N) & (y_N < med_N+alpha*mad_N)
    IN_E = (y_E > med_E-alpha*mad_E) & (y_E < med_E+alpha*mad_E)
    IN = IN_N & IN_E
    del IN_N,IN_E,med_N,mad_N,med_E,mad_E

    # keep selection and get their observation angles
    IN = IN & Msk
    y_N = y_N[IN]
    y_E = y_E[IN]
    y_Zn = abs(obsZn[IN]) # no negative values permitted
    y_Az = obsAz[IN]

    # Generalized Least Squares adjustment, as there might be some heteroskedasticity
    # construct measurement vector and variance matrix
    A = np.tile(np.identity(2, dtype = float), (len(y_N), 1))
    y = np.reshape(np.hstack((y_N[:,np.newaxis],y_E[:,np.newaxis])),(-1,1))

    # # build 2*2*n covariance matrix
    # q_yy = np.zeros((len(y_Zn),2,2), dtype = float)
    # for b in range(len(y_Zn)):
    #     q_yy[b,:,:] = np.matmul(rotMat(y_Az[b]),np.array([[sig_y, 0], [0, (1+np.tan(np.radians(y_Zn[b])))*sig_y]]))
    # # transform to diagonal matrix
    # Q_yy = block_diag(*q_yy)
    # del q_yy,y_N,y_E,y_Zn,y_Az

    xHat = np.linalg.lstsq(A, y, rcond=None)
    xHat = xHat[0]
    # gls_model = sm.GLS(A, y, sigma=Q_yy)
    # x_hat = gls_model.fit()
    Dstack[i,:] = np.transpose(xHat)

xCoreg = np.linalg.lstsq(Astack, Dstack, rcond=None)
xCoreg = xCoreg[0]
# write co-registration
f = open(dat_path+'coreg.txt', 'w')
for i in range(xCoreg.shape[0]):
    line = im_path[i]+' '+'{:+3.4f}'.format(xCoreg[i,0])+' '+'{:+3.4f}'.format(xCoreg[i,1])
    f.write(line + '\n')
f.close()

#lkTransform = RefScale(RefTrans(subTransform,tempSize/2,tempSize/2),tempSize)
#makeGeoIm(Dstack[0],lkTransform,crs,"DispAx1.tif")



# make stack of Labels & Connectivity
for i in range(len(im_path)):
    sen2Path = dat_path + im_path[i]
    # if i==0:
    (Rgi, spatialRef, geoTransform, targetprj) = read_geo_image(sen2Path + 'rgi.tif')
    # img = gdal.Open(sen2Path + 'rgi.tif')
    # Rgi = np.array(img.GetRasterBand(1).ReadAsArray())
        # Cast = np.zeros([Rgi.shape[0],Rgi.shape[1],len(im_path)])
        # Cast = Cast.astype(np.int32)
        # Conn = np.zeros([Rgi.shape[0],Rgi.shape[1],len(im_path)])
        # Conn = Conn.astype(np.int32)
    img = gdal.Open(sen2Path + 'labelPolygons.tif')
    cast = np.array(img.GetRasterBand(1).ReadAsArray())
    img = gdal.Open(sen2Path + 'labelCastConn.tif')
    conn = np.array(img.GetRasterBand(1).ReadAsArray())

    # keep it image-based
    selec = Rgi!=0
    #OBS! use a single RGI entity for debugging
    selec = Rgi==22216
    IN = np.logical_and(selec,conn<0) # are shadow edges on the glacier
    linIdx1 = np.transpose(np.array(np.where(IN)))

    # find the postive number (caster), that is an intersection of lists
    castPtId = conn[IN]
    castPtId *= -1 # switch sign to get caster
    polyId = cast[IN]
    casted = np.transpose(np.vstack((castPtId,polyId))) #.tolist()

    IN = conn>0 # make a selection to reduce the list
    linIdx2 = np.transpose(np.array(np.where(IN)))
    caster = conn[IN]
    polyCa = cast[IN]
    casters = np.transpose(np.vstack((caster,polyCa))).tolist()

    indConn = np.zeros((casted.shape[0]))
    for x in range(casted.shape[0]):
        #idConn = np.where(np.all(casters==casted[x], axis=1))
        try:
            idConn = casters.index(casted[x].tolist())
        except ValueError:
            idConn = -1
        indConn[x] = idConn
    # transform to image coordinates
    # idMated = np.transpose(np.array(np.where(indConn!=-1)))
    OK = indConn!=-1
    idMated = indConn[OK].astype(int)
    castedIJ = linIdx1[OK,:]
    castngIJ = linIdx2[idMated,:]

    # transform to map coordinates
    (castedX,castedY) = pix2map(geoTransform,castedIJ[:,0],castedIJ[:,1])
    (castngX,castngY) = pix2map(geoTransform,castngIJ[:,0],castngIJ[:,1])

    # get sun angles, at casting locations
    (sunZn,sunAz) = read_sun_angles_s2(sen2Path)
    #OBS: still in relative coordinates!!!
    sunZn = sunZn[minI:maxI,minJ:maxJ]
    sunAz = sunAz[minI:maxI,minJ:maxJ]
    sunZen = sunZn[castngIJ[:,0],castngIJ[:,1]]
    sunAzi = sunAz[castngIJ[:,0],castngIJ[:,1]]

    # write to file
    f = open(sen2Path+'conn.txt', 'w')
    for i in range(castedX.shape[0]):
        line = '{:+8.2f}'.format(castngX[i])+' '+'{:+8.2f}'.format(castngY[i])+' '
        line = line + '{:+8.2f}'.format(castedX[i])+' '+'{:+8.2f}'.format(castedY[i])+' '
        line = line + '{:+3.4f}'.format(sunAzi[i])+' '+'{:+3.4f}'.format(sunZen[i])
        f.write(line + '\n')
    f.close()

    del cast,conn

# get co-registration information
with open(dat_path+'coreg.txt') as f:
    lines = f.read().splitlines()
coName = [line.split(' ')[0] for line in lines]
coReg = np.array([list(map(float,line.split(' ')[1:])) for line in lines])
del lines

# construct connectivity
for i in range(GridIdxs.shape[1]):
    fnam1 = im_path[GridIdxs[0][i]]
    fnam2 = im_path[GridIdxs[1][i]]

    # get start and finish points of shadow edges
    conn1 = np.loadtxt(fname = dat_path+fnam1+'conn.txt')
    conn2 = np.loadtxt(fname = dat_path+fnam2+'conn.txt')

    # compensate for coregistration
    coid1 = coName.index(fnam1)
    coid2 = coName.index(fnam2)
    coDxy = coReg[coid1]-coReg[coid2]
    conn2[:,0] += coDxy[0]
    conn2[:,1] += coDxy[1]
    #conn2[:,2] += coDxy[0]
    #conn2[:,3] += coDxy[1]

    # find nearest
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(conn1[:,0:2])
    distances, indices = nbrs.kneighbors(conn2[:,0:2])
    IN = distances<20
    idxConn = np.transpose(np.vstack((np.where(IN)[0], indices[distances<20])))


# walk through glacier polygons
# rgiList = np.trim_zeros(np.unique(Rgi))
# for j in rgiList:
#selection of glacier polygon
    # selec = Rgi==j

# keep it image-based
selec = Rgi!=0

linIdx = np.where(IN)


listOfCoordinates= list(zip(linIdx[0], linIdx[1], linIdx[2]))

# find associated caster
fig, ax = plt.subplots()
im = ax.imshow(selec.astype(int))
fig.colorbar(im)
plt.show()








# weights through observation angle


# makeGeoIm(Mcan,subTransform,crs,"shadowRidges.tif")

# merge lists
for i in range(len(im_path)):
    print('getting together')

#processed_image = cv2.filter2D(image,-1,kernel)




fig, ax = plt.subplots()
im = ax.imshow(Rgi)
fig.colorbar(im)
plt.show()



