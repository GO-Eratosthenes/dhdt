import os
import glob
import math
#import cairo # for drawing of subpixel polygons
import numpy as np

from scipy import ndimage  # for image filtering
from scipy import special  # for trigonometric functions

from skimage import measure
from skimage import segmentation  # for superpixels
from skimage import color  # for labeling image
from skimage.morphology import remove_small_objects # opening, disk, erosion, closing
from skimage.segmentation import active_contour # for snakes
from skimage.filters import threshold_otsu, threshold_multiotsu

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import svm

from rasterio.features import shapes  # for raster to polygon

from shapely.geometry import shape
from shapely.geometry import Point, LineString
from shapely.geos import TopologicalError  # for troubleshooting

from eratosthenes.generic.mapping_io import read_geo_image, read_geo_info
from eratosthenes.generic.mapping_tools import castOrientation, make_same_size
from eratosthenes.generic.mapping_tools import pix_centers, map2pix

from eratosthenes.preprocessing.read_s2 import read_sun_angles_s2

from eratosthenes.processing.handler_s2 import read_mean_sun_angles_s2

def make_shadowing(dem_path, dem_file, im_path, im_name, \
                   Zn=45., Az=-45., nodata=-9999, dtype=bool ):
    
    (dem_mask, crs_dem, geoTransform_dem, targetprj_dem) = read_geo_image(
        os.path.join(dem_path, dem_file))
    if 'MSIL1C' in im_name: # for Sentinel-2
        fname = os.path.join(im_path, im_name, '*B08.jp2')
        (crs_im, geoTransform_im, targetprj_im, rows, cols, bands) = read_geo_info(
            glob.glob(fname)[0])
    else:
        (crs_im, geoTransform_im, targetprj_im, rows, cols, bands) = read_geo_info(
            os.path.join(im_path, im_name))
        
    dem = make_same_size(dem_mask, geoTransform_dem, geoTransform_im, 
                         rows, cols)
    msk = dem==nodata
    
    if 'MSIL1C' in im_name: # for Sentinel-2
        Zn, Az = read_mean_sun_angles_s2(os.path.join(im_path, im_name))
    
    # turn towards the sun
    dem_rot = ndimage.rotate(dem, Az, 
                             axes=(1, 0), reshape=True, output=None, 
                             order=3, mode='constant', cval=nodata)
    msk_rot = ndimage.rotate(msk, Az, 
                             axes=(1, 0), reshape=True, output=None, 
                             order=0, mode='constant', cval=True)
    del dem
    shw_rot = np.zeros_like(dem_rot, dtype=bool)
    
    dZ = special.tandg(90-Zn) * np.sqrt(1 + 
                                        min(special.tandg(Az)**2,
                                            special.cotdg(Az)**2)) * geoTransform_dem[1]
    
    cls_rot = np.bitwise_not(ndimage.binary_dilation(msk_rot, 
                                                    structure=np.ones((7,7)))
                             ).astype(int)
    # sweep over matrix
    for swp in range(1,dem_rot.shape[0]):
        shw_rot[swp,:] = (dem_rot[swp,:] < dem_rot[swp-1,:]-dZ) & ( cls_rot[swp,:]==1 )
        dem_rot[swp,:] = np.maximum((dem_rot[swp,:]*cls_rot[swp,:]), 
                                    (dem_rot[swp-1,:]*cls_rot[swp-1,:]) - dZ*cls_rot[swp,:])
    del dem_rot, cls_rot
    msk_new = ndimage.rotate(msk_rot, -Az, 
                             axes=(1, 0), reshape=True, output=None, 
                             order=0, mode='constant', cval=True)
        
    i_min = int(np.floor((msk_new.shape[0] - msk.shape[0])/2))
    i_max = int(np.floor((msk_new.shape[0] + msk.shape[0])/2))
    j_min = int(np.floor((msk_new.shape[1] - msk.shape[1])/2))
    j_max = int(np.floor((msk_new.shape[1] + msk.shape[1])/2))
    
    
    if isinstance(dtype, (int, float)):
        shw_rot.astype(dtype)
        shw_new = ndimage.rotate(shw_rot, -Az, 
                             axes=(1, 0), reshape=True, output=None, 
                             order=1, mode='constant', cval=False) 
    else:    
        shw_new = ndimage.rotate(shw_rot, -Az, 
                             axes=(1, 0), reshape=True, output=None, 
                             order=0, mode='constant', cval=False)   

    Shw = shw_new[i_min:i_max, j_min:j_max]
    return Shw 

def make_shading(dem_path, dem_file, im_path, im_name, \
                 Zn=np.radians(45.), Az=np.radians(-45.), nodata=-9999,):
    """
    make hillshading from elevation model and meta data of image file

    """
    (dem_mask, crs_dem, geoTransform_dem, targetprj_dem) = read_geo_image(
        os.path.join(dem_path, dem_file))
    if 'MSIL1C' in im_name: # for Sentinel-2
        fname = os.path.join(im_path, im_name, '*B08.jp2')
        (crs_im, geoTransform_im, targetprj_im, rows, cols, bands) = read_geo_info(
            glob.glob(fname)[0])
    else:
        (crs_im, geoTransform_im, targetprj_im, rows, cols, bands) = read_geo_info(
            os.path.join(im_path, im_name))
    
    dem = make_same_size(dem_mask, geoTransform_dem, geoTransform_im, 
                         rows, cols)
    
    if 'MSIL1C' in im_name: # for Sentinel-2
        Zn, Az = read_mean_sun_angles_s2(os.path.join(im_path, im_name))
        Zn = np.radians(Zn)
        Az = np.radians(Az)

    
    # convert to unit vectors
    sun = np.dstack((np.sin(Az), np.cos(Az), np.tan(Zn)))
    n = np.linalg.norm(sun, axis=2)
    sun[:, :, 0] /= n
    sun[:, :, 1] /= n
    sun[:, :, 2] /= n
    
    # estimate surface normals
    dy, dx = np.gradient(dem*geoTransform_im[1])  

    normal = np.dstack((-dx, dy, np.ones_like(dem)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    Shd = normal[:,:,0]*sun[:,:,0] + normal[:,:,1]*sun[:,:,1] + normal[:,:,2]*sun[:,:,2]
    return Shd

def create_shadow_polygons(M, im_path, bbox=(0, 0, 0, 0), \
                           median_filtering=True):
    """
    Generate polygons from floating array, combine this with sun illumination
    information, to generate connected pixels on the edge of these polygons
    input:   M              array (m x n)     array with intensity values
             im_path        string            location of the image metadata
    output:  labels         array (m x n)     array with numbered labels
             label_ridge    array (m x n)     array with numbered superpixels
             cast_conn      array (m x n)     array with numbered edge pixels
    """
    
    minI = bbox[0] 
    maxI = bbox[1] 
    minJ = bbox[2]
    maxJ = bbox[3] 
    
    if median_filtering:
        siz, loop = 5, 10
        M = median_filter_shadows(M,siz,loop)
        
    labels,thres_val = sturge(M) # classify into regions

    # get self-shadow and cast-shadow
    (sunZn,sunAz) = read_sun_angles_s2(im_path)
    if (minI!=0 or maxI!=0 and minI!=0 or maxI!=0):
        sunZn = sunZn[minI:maxI,minJ:maxJ]
        sunAz = sunAz[minI:maxI,minJ:maxJ]  
    
    cast_conn = labelOccluderAndCasted(labels, sunAz, M)#, subTransform)
    
    return labels, cast_conn

# geometric functions
def getShadowPolygon(M, sizPix, thres):  # pre-processing
    """
    Use superpixels to group and label an image
    input:   M              array (m x n)     array with intensity values
             sizPix         integer           window size of the kernel
             thres          integer           threshold value for
                                              Region Adjacency Graph
    output:  labels         array (m x n)     array with numbered labels
             SupPix         array (m x n)     array with numbered superpixels
    """

    mn = np.ceil(np.divide(np.nanprod(M.shape), sizPix))
    SupPix = segmentation.slic(M, sigma=1,
                               n_segments=mn,
                               compactness=0.010)  # create super pixels

    #    g = graph.rag_mean_color(M, SupPix) # create region adjacency graph
    #    mc = np.empty(len(g))
    #    for n in g:
    #        mc[n] = g.nodes[n]['mean color'][1]
    #    graphCut = graph.cut_threshold(SupPix, g, thres)
    #    meanIm = color.label2rgb(graphCut, M, kind='avg')
    meanIm = color.label2rgb(SupPix, M, kind='avg')
    sturge = 1.6 * (math.log2(mn) + 1)
    values, base = np.histogram(np.reshape(meanIm, -1),
                                bins=np.int(np.ceil(sturge)))
    dips = findValley(values, base, 2)
    val = max(dips)
    #    val = filters.threshold_otsu(meanIm)
    #    val = filters.threshold_yen(meanIm)
    imSeparation = meanIm > val
    labels = measure.label(imSeparation, background=0)
    labels = np.int16(
        labels)  # so it can be used for the boundaries extraction
    return labels, SupPix

# image enhancement
def median_filter_shadows(M, siz, loop):
    """
    Transform intensity to more clustered intensity, through iterative
    filtering with a median operation
    input:   M              array (m x n)     array with intensity values
             siz            integer           window size of the kernel
             loop           integer           number of iterations
    output:  Mmed           array (m x n)     array with stark edges
    """
    for i in range(loop):
        M = ndimage.median_filter(M, size=siz)
    return M

def mean_shift_filter(M, quan=0.1, n=1000):
    """
    Transform intensity to more clustered intensity, through mean-shift
    filtering
    input:   M              array (m x n)     array with intensity values
             quantile       float             
             n              integer           
    output:  Mmed           array (m x n)     array with stark edges
    """
    (m,n) = M.shape
    bw = estimate_bandwidth(M.reshape(-1,1), quantile=quan, n_samples=n)    
    ms = MeanShift(bandwidth=bw, bin_seeding=True)
    ms.fit(M.reshape(-1,1))

    labels = np.reshape(ms.labels_, (m,n))
    return labels

# threshholding and classification functions
def sturge(M):
    """
    Transform intensity to labelled image
    input:   M              array (m x n)     array with intensity values
    output:  labels         array (m x n)     array with numbered labels
    """
    mn = M.size
    sturge = 1.6 * (math.log2(mn) + 1)
    values, base = np.histogram(np.reshape(M, -1),
                                bins=np.int(np.ceil(sturge)))
    dips = findValley(values, base, 2)
    val = max(dips)
    imSeparation = M > val
    
    # remove "salt and pepper" noise
    imSeparation = remove_small_objects(imSeparation, min_size=10)
    
    labels, num_polygons = ndimage.label(imSeparation)
    return labels, val

def findValley(values, base, neighbors=2):
    """
    A valley is a point which has "n" consequative higher values on both sides
    input:   values         array (m x 1)     vector with number of occurances
             base           array (m x 1)     vector with central values
             neighbors      integer           number of neighbors needed
                                              in order to be a valley
    output:  dips           array (n x 1)     array with valley locations
    """
    for i in range(neighbors):
        if i == 0:
            wallP = np.roll(values, +(i + 1))
            wallM = np.roll(values, -(i + 1))
        else:
            wallP = np.vstack((wallP, np.roll(values, +(i + 1))))
            wallM = np.vstack((wallM, np.roll(values, -(i + 1))))
    if neighbors > 1:
        concavP = np.all(np.sign(np.diff(wallP, n=1, axis=0)) == +1, axis=0)
        concavM = np.all(np.sign(np.diff(wallM, n=1, axis=0)) == +1, axis=0)
    selec = np.all(np.vstack((concavP, concavM)), axis=0)
    selec = selec[neighbors - 1:-(neighbors + 1)]
    idx = base[neighbors - 1:-(neighbors + 2)]
    dips = idx[selec]
    # walls = np.amin(sandwich,axis=0)
    # selec = values<walls
    return dips

def shadow_image_to_list(M, geoTransform, sen2Path, inputArg, 
                         method='otsu', Shw=0., Zn=0., Az=0.):
    '''
    Turns the image towards the sun, and looks along each solar ray
    
    
    :param M:         NP.ARRAY (_,_)
        shadow enhanced imagery
    :param I2:        LIST (1,6)
        GDAL georeference structure
    :param sen2path:  STRING
        path where sentine
    :param sen2path:  DICTONARY
        user inputs
    :param iteration: INTEGER
        number of iterations used
    :param iteration: INTEGER
        number of iterations used
                     
    :return u:        FLOAT
        displacement estimate
    :return v:        FLOAT     
        displacement estimate                    
    '''
    if method=='otsu':
        Zn, Az = read_mean_sun_angles_s2(sen2Path)
    
    # turn towards the sun
    # shade_thre = threshold_multiotsu(M, classes=3, nbins=1000)
    M_rot = ndimage.rotate(M, 180-Az, 
                           axes=(1, 0), reshape=True, output=None, 
                           order=3, mode='constant', cval=-9999)
    
    if method=='svm':
        S_rot = ndimage.rotate(Shw, 180-Az, 
                               axes=(1, 0), reshape=True, output=None, 
                               order=0, mode='constant', cval=-9999)    
    
    # tform = transform.SimilarityTransform(translation=(0, -10)) #rotation=Az*(np.pi/180))
    # M_rot = transform.warp(M, tform)
    #
    # neither ndimage or transform.warp gives the new coordinate system.
    # hence, clumpsy interpolation of the grid is applied here
    X,Y = pix_centers(geoTransform, M.shape[0], M.shape[1], make_grid=True)
    X_rot = ndimage.rotate(X, 180-Az,
                           axes=(1, 0), reshape=True, output=None, 
                           order=3, mode='constant', cval=X[0,0])# 
    Y_rot = ndimage.rotate(Y, 180-Az,
                           axes=(1, 0), reshape=True, output=None, 
                           order=3, mode='constant', cval=-9999)# Y[0,0]
    
    suntrace_list = np.empty((0, 4))
    for k in range(M_rot.shape[1]):
        M_trace = M_rot[:,k] # line of intensities along a sun trace
        IN = M_trace!=-9999
        
        if method=='otsu':
            shade_thre = threshold_multiotsu(M_trace[IN], 
                                             classes=3, 
                                             nbins=np.sum(IN)//10)
        else: # support vector machine
            S_trace = S_rot[:,k] # line of intensities along a sun trace
            clf = svm.SVC(kernel='linear')
            clf.fit(np.vstack((M_trace[IN], 0*IN[IN])).T, 
                    2*(S_trace[IN])-1)
            w = clf.coef_[0]
            shade_thre = -clf.intercept_[0]/w[0]

        shade_class = M_trace>shade_thre[1] # take the upper threshold
        shade_class = shade_class.astype(int)
        shade_exten = np.pad(shade_class, (1,1), 'constant', 
                             constant_values=0)
        shade_node = np.roll(shade_exten, 1)-shade_exten
    
        # (col_idx, ) = np.where(IN)
        if -90<Az<90:
            (shade_beg, ) = np.where(shade_node[1::]==-1)
            (shade_end, ) = np.where(shade_node[2::]==+1)
        else:
            (shade_beg, ) = np.where(shade_node[2::]==+1)
            (shade_end, ) = np.where(shade_node[1::]==-1)
        
        if len(shade_beg)!=0:
            # coordinate transform, not image transform (loss of points)
            col_idx = k*np.ones(shade_beg.size, dtype=np.int32)
            x_beg = X_rot[shade_beg, col_idx] #index seems shifted...?
            x_end = X_rot[shade_end, col_idx]
            y_beg = Y_rot[shade_beg, col_idx]
            y_end = Y_rot[shade_end, col_idx]

            suntrace_list = np.vstack((suntrace_list , 
                                       np.transpose(np.vstack(
                                           (x_beg[np.newaxis], 
                                            y_beg[np.newaxis], 
                                            x_end[np.newaxis], 
                                            y_end[np.newaxis] )) ) ))
        
    # remove out of image sun traces
    OUT = np.any(suntrace_list==-9999, axis=1)
    
    # find azimuth and elevation at cast location
    (sunZn,sunAz) = read_sun_angles_s2(sen2Path)
    if inputArg['bbox'] is not None:
        sunZn = sunZn[inputArg['bbox'][0]:inputArg['bbox'][1],
                      inputArg['bbox'][2]:inputArg['bbox'][3]]
        sunAz = sunAz[inputArg['bbox'][0]:inputArg['bbox'][1],
                      inputArg['bbox'][2]:inputArg['bbox'][3]]
    (iC,jC) = map2pix(geoTransform, 
                      suntrace_list[:,0].copy(), suntrace_list[:,1].copy())
    iC, jC = np.round(iC), np.round(jC)
    iC, jC = iC.astype(int), jC.astype(int)
    IN = (0>=iC) & (iC<=sunZn.shape[0]) & (0>=jC) & (jC<=sunZn.shape[1])
    iC[~IN] = 0 
    jC[~IN] = 0 
    sun_angles = np.transpose(np.vstack((sunAz[iC,jC], sunZn[iC,jC])))
    
    # write to file
    f = open(sen2Path + 'conn.txt', 'w')
    
    for k in range(suntrace_list.shape[0]):
        line = '{:+8.2f}'.format(suntrace_list[k,0]) + ' '
        line += '{:+8.2f}'.format(suntrace_list[k,1]) + ' '
        line += '{:+8.2f}'.format(suntrace_list[k,2]) + ' '
        line += '{:+8.2f}'.format(suntrace_list[k,3]) + ' '
        line += '{:+3.4f}'.format(sun_angles[k,0]) + ' ' 
        line += '{:+3.4f}'.format(sun_angles[k,1])
        f.write(line + '\n')
    f.close()

def labelOccluderAndCasted(labeling, sunAz, M=None):  # pre-processing
    """
    Find along the edge, the casting and casted pixels of a polygon
    input:   labeling       array (n x m)     array with labelled polygons
             sunAz          array (n x m)     band of azimuth values
             M              array (n x m)     intensity image of the shadow
    output:  shadowIdx      array (m x m)     array with numbered pairs, where
                                              the caster is the positive number
                                              the casted is the negative number
    """
    labeling = labeling.astype(np.int64)
    msk = labeling >= 1
    mL, nL = labeling.shape
    shadowIdx = np.zeros((mL, nL), dtype=np.int16)
    # shadowRid = np.zeros((mL,nL), dtype=np.int16)
    inner = ndimage.morphology.binary_erosion(msk)
    # inner = ndimage.morphology.binary_dilation(msk==0)&msk
    bndOrient = castOrientation(inner.astype(np.float), sunAz)
    del mL, nL, inner

    # labelList = np.unique(labels[msk])
    # locs = ndimage.find_objects(labeling, max_label=0)

    labList = np.unique(labeling)
    labList = labList[labList != 0]
    for i in labList:
        selec = labeling == i
        labIdx = np.nonzero(selec)
        labImin = np.min(labIdx[0])
        labImax = np.max(labIdx[0])
        labJmin = np.min(labIdx[1])
        labJmax = np.max(labIdx[1])
        subMsk = selec[labImin:labImax, labJmin:labJmax]

        # for i,loc in enumerate(locs):
        #     subLabel = labels[loc]

        # for i in range(len(locs)): # range(1,labels.max()): # loop through
        #                                                     # all polygons
        #    # print("Starting on number %s" % (i))
        #    loc = locs[i]
        #    if loc is not None:
        #        subLabel = labels[loc] # generate subset covering
        #                               # only the polygon
        # # slices seem to be coupled.....
        #        subLabel = labels[loc[0].start:loc[0].stop,
        #                          loc[1].start:loc[1].stop]
        #    subLabel[subLabel!=(i+0)] = 0   # de boosdoener
        subOrient = np.sign(bndOrient[labImin:labImax,
                            labJmin:labJmax])
        # subOrient = np.sign(bndOrient[loc])

        # subMsk = subLabel==(i+0)
        # subBound = subMsk & ndimage.morphology.binary_dilation(subMsk==0)
        subBound = subMsk ^ ndimage.morphology.binary_erosion(subMsk)
        subOrient[~subBound] = 0  # remove other boundaries

        subAz = sunAz[labImin:labImax,
                      labJmin:labJmax]  # [loc] # subAz = sunAz[loc]

        subWhe = np.nonzero(subMsk)
        ridgIdx = subOrient[subWhe[0], subWhe[1]] == 1
        try:
            ridgeI = subWhe[0][ridgIdx]
            ridgeJ = subWhe[1][ridgIdx]
        except IndexError:
            continue
            # try:
            #     shadowRid[labIdx] = i
            #     # shadowRid[loc] = i
            #     # shadowRid[ridgeI+(loc[0].start),ridgeJ+(loc[1].start)] = i
        # except IndexError:
        #     print('iets aan de hand')

        # boundary of the polygon that receives cast shadow
        cast = subOrient == -1

        m, n = subMsk.shape
        print("For shadowpolygon #%s: Its size is %s by %s, connecting %s pixels in total"
              % (i, m, n, len(ridgeI)))

        for x in range(len(ridgeI)):  # loop through all occluders
            sunDir = subAz[ridgeI[x]][ridgeJ[x]]  # degrees [-180 180]

            # Bresenham's line algorithm
            dI = -math.cos(math.radians(subAz[ridgeI[x]][ridgeJ[
                x]]))  # -cos # flip axis to get from world into image coords
            dJ = -math.sin(math.radians(subAz[ridgeI[x]][ridgeJ[x]]))  # -sin #
            brd = 3 # add aditional cast borders to the suntrace
            if abs(sunDir) > 90:  # northern hemisphere
                if dI > dJ:
                    rr = np.arange(start=0, stop=ridgeI[x]+brd, step=1)
                    cc = np.flip(np.round(rr * dJ), axis=0) + ridgeJ[x]
                else:
                    cc = np.arange(start=0, stop=ridgeJ[x]+brd, step=1)
                    rr = np.flip(np.round(cc * dI), axis=0) + ridgeI[x]
            else:  # southern hemisphere
                if dI > dJ:
                    rr = np.arange(start=ridgeI[x]-brd, stop=m, step=1)
                    cc = np.round(rr * dJ) + ridgeJ[x]
                else:
                    cc = np.arange(start=ridgeJ[x]-brd, stop=n, step=1)
                    rr = np.round(cc * dI) + ridgeI[x]
                    # generate cast line in sub-image
            rr = rr.astype(np.int64)
            cc = cc.astype(np.int64)
            IN = (cc >= 0) & (cc <= n) & (rr >= 0) & (
                            rr <= m)  # inside sub-image
            if IN.any():
                rr = rr[IN]
                cc = cc[IN]
                
                if M is not None: # use intensity data
                    # find transitions between illuminated and shadowed pixels along a line
                    shade_line = M[rr,cc]
                    if np.std(shade_line) == 0: # if no transition
                        shade_thre = shade_line[0]
                    else:
                        shade_thre = threshold_otsu(shade_line)
                    shade_class = shade_line>=shade_thre
                    shade_class = shade_class.astype(int)
                    shade_exten = np.pad(shade_class, (1,1), 'constant', 
                                         constant_values=0)
                    shade_node = np.roll(shade_exten, 1)-shade_exten
                    if -90<sunDir<90:
                        (shade_beg, ) = np.where(shade_node[1::]==-1)
                        (shade_end, ) = np.where(shade_node[2::]==+1)
                    else:
                        (shade_beg, ) = np.where(shade_node[2::]==+1) 
                        (shade_end, ) = np.where(shade_node[1::]==-1)
                        
                    rrCast, ccCast = rr[shade_beg], cc[shade_beg]
                    rrShad, ccShad = rr[shade_end], cc[shade_end]
                    
                    ridgeI[x]
                    ### OBS: almost right track...
                    
                else:
                    subCast = np.zeros((m, n), dtype=np.uint8)
                    try:
                        subCast[rr, cc] = 1
                    except IndexError:
                        continue
                    
                        # find closest casted
                    castedHit = cast & subCast
                    (castedIdx) = np.where(castedHit[subWhe[0], subWhe[1]] == 1)
                    castedI = subWhe[0][castedIdx[0]]
                    castedJ = subWhe[1][castedIdx[0]]
                    del IN, castedIdx, castedHit, subCast, rr, cc, dI, dJ, sunDir
    
                    if len(castedI) > 1:
                        # do selection of the closest casted
                        dist = np.sqrt(
                            (castedI - ridgeI[x]) ** 2 + (castedJ - ridgeJ[x]) ** 2)
                        idx = np.where(dist == np.amin(dist))
                        castedI = castedI[idx[0]]
                        castedJ = castedJ[idx[0]]
    
                    if len(castedI) > 0:
                        # write out
                        shadowIdx[ridgeI[x] + labImin][ridgeJ[x] + labJmin] = +(x+1)
                        # shadowIdx[ridgeI[x]+loc[0].start,
                        #           ridgeJ[x]+loc[1].start] = +x # ridge
                        shadowIdx[castedI[0] + labImin][castedJ[0] + labJmin] = -(x+1)
        
        print("polygon done")
                    # shadowIdx[castI[0]+loc[0].start,
                    #           castJ[0]+loc[1].start] = -x # casted
            #     else:
            #         print('out of bounds')
            # else:
            #     print('out of bounds')

            # fssubShadowIdx = shadowIdx[labImin:labImax, labJmin:labJmax]
            # subShadowIdx = shadowIdx[loc]

            # subShadowOld = shadowIdx[loc]  # make sure other
            #                                # are not overwritten
            # OLD = subShadowOld!=0
            # subShadowIdx[OLD] = subShadowOld[OLD]
            # shadowIdx[loc] = subShadowIdx
    return shadowIdx


def listOccluderAndCasted(labels, sunZn, sunAz,
                          geoTransform):  # pre-processing
    """
    Find along the edge, the casting and casted pixels of a polygon
    input:   labels         array (n x m)     array with labelled polygons
             sunAz          array (n x m)     band of azimuth values
             sunZn          array (n x m)     band of zentih values
    output:  castList       list  (k x 6)     array with image coordinates of
                                              caster and casted with the
                                              sun angles of the caster
    """
    msk = labels > 1
    labels = labels.astype(np.int32)
    mskOrient = castOrientation(msk.astype(np.float), sunAz)
    mskOrient = np.sign(mskOrient)
    # makeGeoIm(mskOrient,subTransform,crs,"polyRidges.tif")

    castList = []
    for shp, val in shapes(labels, mask=msk, connectivity=8):
        #        coord = shp["coordinates"]
        #        coord = np.uint16(np.squeeze(np.array(coord[:])))
        if val != 0:
            #    if val==48:
            # get ridge coordinates
            polygoon = shape(shp)
            polyRast = labels == val  # select the polygon
            polyInnr = ndimage.binary_erosion(polyRast,
                                              np.ones((3, 3),
                                                      dtype=bool))
            polyBoun = np.logical_xor(polyRast, polyInnr)
            polyWhe = np.nonzero(polyBoun)
            ridgIdx = mskOrient[polyWhe[0], polyWhe[1]] == 1
            ridgeI = polyWhe[0][ridgIdx]
            ridgeJ = polyWhe[1][ridgIdx]
            del polyRast, polyInnr, polyBoun, polyWhe, ridgIdx

            for x in range(len(ridgeI)):  # ridgeI:
                try:
                    castLine = LineString([[ridgeJ[x], ridgeI[x]],
                                           [ridgeJ[x]
                                            - (math.sin(math.radians(
                                               sunAz[ridgeI[x]][
                                                   ridgeJ[x]])) * 1e4),
                                            ridgeI[x]
                                            + (math.cos(math.radians(
                                                sunAz[ridgeI[x]][
                                                    ridgeJ[x]])) * 1e4)]])
                except IndexError:
                    continue
                try:
                    castEnd = polygoon.intersection(castLine)
                except TopologicalError:
                    # somehow the exterior of the polygon crosses or touches
                    # itself, making it a LinearRing
                    polygoon = polygoon.buffer(0)
                    castEnd = polygoon.intersection(castLine)

                if castEnd.geom_type == 'LineString':
                    castEnd = castEnd.coords[:]
                elif castEnd.geom_type == 'MultiLineString':
                    # castEnd = [list(x.coords) for x in list(castEnd)]
                    cEnd = []
                    for m in list(castEnd):
                        cEnd += m.coords[:]
                    castEnd = cEnd
                    del m, cEnd
                elif castEnd.geom_type == 'GeometryCollection':
                    cEnd = []
                    for m in range(len(castEnd)):
                        cEnd += castEnd[m].coords[:]
                    castEnd = cEnd
                    del m, cEnd
                elif castEnd.geom_type == 'Point':
                    castEnd = []
                else:
                    print('something went wrong?')

                # if empty
                if len(castEnd) > 1:
                    # if len(castEnd.coords[:])>1:
                    # find closest intersection
                    occluder = Point(ridgeJ[x], ridgeI[x])
                    # dists = [Point(c).distance(occluder)
                    #          for c in castEnd.coords]
                    dists = [Point(c).distance(occluder) for c in castEnd]
                    dists = [float('Inf') if i == 0 else i for i in dists]
                    castIdx = dists.index(min(dists))
                    casted = castEnd[castIdx]

                    # transform to UTM and append to array
                    ridgeX = (ridgeI[x] * geoTransform[2]
                              + ridgeJ[x] * geoTransform[1]
                              + geoTransform[0]
                              )
                    ridgeY = (ridgeI[x] * geoTransform[5]
                              + ridgeJ[x] * geoTransform[4]
                              + geoTransform[3]
                              )
                    castX = (casted[1] * geoTransform[2]
                             + casted[0] * geoTransform[1]
                             + geoTransform[0]
                             )
                    castY = (casted[1] * geoTransform[5]
                             + casted[0] * geoTransform[4]
                             + geoTransform[3]
                             )

                    castLine = np.array([ridgeX, ridgeY, castX, castY,
                                         sunAz[ridgeI[x]][ridgeJ[x]],
                                         sunZn[ridgeI[x]][ridgeJ[x]]])
                    castList.append(castLine)
                    del dists, occluder, castIdx, casted
                del castLine, castEnd
    return castList

# # shadow imagery at sub-pixel resolution
# def draw_shadow_polygons(path_shadow, path_label, alpha, beta, gamma):
#     (M, crs, geoTransform, targetprj) = read_geo_image(path_shadow)
#     (labels, crs, geoTransform, targetprj) = read_geo_image(path_label)

#     surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 
#                                  np.shape(M)[1], np.shape(M)[0])
#     ctx = cairo.Context(surface)   
#     # run through all polygons, one per one
#     for i in np.arange(1,np.max(labels)+1):           
#         # create closed polygon
#         subLabel = labels==i
#         subLabel = ndimage.morphology.binary_dilation(subLabel, 
#                                                       ndimage.generate_binary_structure(2, 2))
#         contours = measure.find_contours(subLabel, level=0.5, 
#                                         fully_connected='high', 
#                                         positive_orientation='low', mask=None)
#         for j in range(len(contours)):
#             contour = contours[j]
#             # make a snake
#             snake = active_contour(M,
#                            np.vstack((contour,contour[0,:])), 
#                            alpha=alpha, beta=beta, gamma=gamma, 
#                            w_edge=+5, coordinates='rc',
#                            max_iterations=5)

#             ctx.move_to(snake[0,1], snake[0,0])
#             for k in np.arange(1,len(snake)):
#                 ctx.line_to(snake[k,1], snake[k,0])
#             ctx.close_path()
#             ctx.set_source_rgb(1, 1, 1)
#             ctx.fill()

#     buf = surface.get_data()
#     Msnk = np.ndarray(shape=np.shape(M), dtype=np.uint32, buffer=buf)/2**32
#     return Msnk, geoTransform, crs 

    
    
    