from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
import osr
import ogr
import gdal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio
import glob
import random # for PCA

from rasterio.plot import show
from rasterio.mask import mask
from rasterio.features import shapes # for raster to polygon

from shapely.geometry import mapping
from shapely.geometry import shape
from shapely.geometry import Point, LineString

from shapely.geos import TopologicalError # for troubleshooting

import math

from xml.etree import ElementTree
# from xml.dom import minidom
from scipy.interpolate import griddata #for grid interpolation
from scipy import ndimage # for image filtering
from scipy import signal # for convolution filter
from scipy.linalg import block_diag # for co-variance matrix

import statsmodels.api as sm # for generalized least squares

#import cv2

from skimage import measure
#from skimage import draw
#from skimage import graph 
from skimage.future import graph # for rag
from skimage import segmentation # for superpixels
from skimage import color # for labeling image
from skimage import filters # for Otsu thesholding
from skimage import transform # for rotation

from itertools import compress # for fast indexing

def read_band_image(band, path):
    """
    This function takes as input the Sentinel-2 band name and the path of the 
    folder that the images are stored, reads the image and returns the data as
    an array
    input:   band           string            Sentinel-2 band name
             path           string            path of the folder
    output:  data           array (n x m)     array of the band image
             spatialRef     string            projection 
             geoTransform   tuple             affine transformation coefficients
             targetprj                        spatial reference
    """
    fname = os.path.join(path,'*B'+band+'.jp2')
    img = gdal.Open(glob.glob(fname)[0])
    data = np.array(img.GetRasterBand(1).ReadAsArray())
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt = img.GetProjection())
    return data, spatialRef, geoTransform, targetprj

def read_sun_angles(path):
    """
    This function reads the xml-file of the Sentinel-2 scene and extracts an
    array with sun angles, as these vary along the scene.
    """
    fname = os.path.join(path,'MTD_TL.xml')
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()

    # image dimensions
    for meta in root.iter('Size'):
        res = float(meta.get('resolution'))
        if res==10: # take 10 meter band
            mI = float(meta[0].text)
            nI = float(meta[1].text)
    
    # get Zenith array
    Zenith = root[1][1][0][0][2]
    Zn = get_array_from_xml(Zenith)
    znSpac = float(root[1][1][0][0][0].text)
    znSpac = np.stack((znSpac, float(root[1][1][0][0][1].text)),0)
    znSpac = np.divide(znSpac,10) # transform from meters to pixels
    
    zi = np.linspace(0-10,mI+10,np.size(Zn,axis=0))
    zj = np.linspace(0-10,nI+10,np.size(Zn,axis=1))
    Zi,Zj = np.meshgrid(zi,zj)
    Zij = np.dstack([Zi, Zj]).reshape(-1, 2)
    del zi,zj,Zi,Zj,znSpac
    
    iGrd = np.arange(0,mI)
    jGrd = np.arange(0,nI)
    Igrd, Jgrd = np.meshgrid(iGrd, jGrd)

    Zn = griddata(Zij,Zn.reshape(-1),(Igrd,Jgrd), method="linear")
    
    # get Azimuth array
    Azimuth = root[1][1][0][1][2]
    Az = get_array_from_xml(Azimuth)  
    azSpac = float(root[1][1][0][1][0].text)
    azSpac = np.stack((azSpac, float(root[1][1][0][1][1].text)),0)  

    ai = np.linspace(0-10,mI+10,np.size(Az,axis=0))
    aj = np.linspace(0-10,nI+10,np.size(Az,axis=1))
    Ai,Aj = np.meshgrid(ai,aj)
    Aij = np.dstack([Ai, Aj]).reshape(-1, 2)
    del ai,aj,Ai,Aj,azSpac
    
    Az = griddata(Aij,Az.reshape(-1),(Igrd,Jgrd), method="linear")
    del Igrd,Jgrd,Zij,Aij 
    return Zn, Az

def read_view_angles(path):
    """
    This function reads the xml-file of the Sentinel-2 scene and extracts an
    array with sun angles, as these vary along the scene.
    """
    fname = os.path.join(path,'MTD_TL.xml')
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()

    # image dimensions
    for meta in root.iter('Size'):
        res = float(meta.get('resolution'))
        if res==10: # take 10 meter band
            mI = float(meta[0].text)
            nI = float(meta[1].text)
    # get coarse grids
    for grd in root.iter('Viewing_Incidence_Angles_Grids'):
        bid = float(grd.get('bandId'))
        if bid==4: # take band 4
            Zarray = get_array_from_xml(grd[0][2])
            Aarray = get_array_from_xml(grd[1][2])
            if 'Zn' in locals():
                Zn = np.nanmean(np.stack((Zn,Zarray),axis=2), axis=2)
                Az = np.nanmean(np.stack((Az,Aarray),axis=2), axis=2)
            else:
                Zn = Zarray
                Az = Aarray
            del Aarray,Zarray
            
    # upscale to 10 meter resolution    
    zi = np.linspace(0-10,mI+10,np.size(Zn,axis=0))
    zj = np.linspace(0-10,nI+10,np.size(Zn,axis=1))
    Zi,Zj = np.meshgrid(zi,zj)
    Zij = np.dstack([Zi, Zj]).reshape(-1, 2)
    del zi,zj,Zi,Zj
    
    iGrd = np.arange(0,mI)
    jGrd = np.arange(0,nI)
    Igrd, Jgrd = np.meshgrid(iGrd, jGrd)
    Zok = ~np.isnan(Zn)
    Zn = griddata(Zij[Zok.reshape(-1),:],Zn[Zok],(Igrd,Jgrd), method="linear")
    
    ai = np.linspace(0-10,mI+10,np.size(Az,axis=0))
    aj = np.linspace(0-10,nI+10,np.size(Az,axis=1))
    Ai,Aj = np.meshgrid(ai,aj)
    Aij = np.dstack([Ai, Aj]).reshape(-1, 2)
    del ai,aj,Ai,Aj
    
    Aok = ~np.isnan(Az) # remove NaN values from interpolation
    Az = griddata(Aij[Aok.reshape(-1),:],Az[Aok],(Igrd,Jgrd), method="linear")
    del Igrd,Jgrd,Zij,Aij 
    return Zn, Az

def get_array_from_xml(treeStruc):
    """
    Arrays within a xml structure are given line per line
    Output is an array
    """
    for i in range(0, len(treeStruc)):
        Trow = [float(s) for s in treeStruc[i].text.split(' ')]
        if i==0:
            Tn = Trow
        elif i==1:
            Trow = np.stack((Trow, Trow),0)
            Tn = np.stack((Tn, Trow[1,:]),0)
        else:
            # Trow = np.stack((Trow, Trow),0)
            # Tn = np.concatenate((Tn, [Trow[1,:]]),0)
            Tn = np.concatenate((Tn, [Trow]),0)
    return Tn

# shadow functions
def rgb2hsi(R,G,B):
    """
    Transform Red Green Blue arrays to Hue Saturation Intensity arrays
    """
    Red = np.float64(R)/2**16
    Green = np.float64(G)/2**16
    Blue = np.float64(B)/2**16
    
    Hue = np.copy(Red)
    Sat = np.copy(Red)
    Int = np.copy(Red)
    
    # See Tsai, IEEE Trans. Geo and RS, 2006.
    # from Pratt, 1991, Digital Image Processing, Wiley
    Tsai = np.array([(1/3,1/3,1/3), \
                     (-math.sqrt(6)/6,-math.sqrt(6)/6,-math.sqrt(6)/3), \
                     (1/math.sqrt(6),2/-math.sqrt(6),0)])
    
    for i in range(0, Red.shape[0]):
        for j in range(0, Red.shape[1]):
            hsi = np.matmul(Tsai, np.array([ [Red[i][j]], [Green[i][j]] , [Blue[i][j]] ]))
            Int[i][j] = hsi[0]
            Sat[i][j] = math.sqrt(hsi[1]**2 + hsi[2]**2)
            Hue[i][j] = math.atan2(hsi[1],hsi[2])
            
    return Hue, Sat, Int

def nsvi(B2,B3,B4):
    """
    Transform Red Green Blue arrays to normalized saturation-value difference index (NSVI)
    following Ma et al. 2008
    """
    (H,S,I) = rgb2hsi(B4,B3,B2) # create HSI bands
    NSVDI = (S-I)/(S+I)
    return NSVDI

def mpsi(B2,B3,B4):
    """
    Transform Red Green Blue arrays to Mixed property-based shadow index (MPSI)
    following Han et al. 2018
    """
    Green = np.float64(B3)/2**16
    Blue = np.float64(B2)/2**16
    
    (H,S,I) = rgb2hsi(B4,B3,B2) # create HSI bands
    MPSI = (H-I)*(Green-Blue)
    return MPSI

def pca(X):
    """
    principle component analysis (PCA)
    """
    # Data matrix X, assumes 0-centered
    n, m = X.shape
# removed, because a sample is taken:    
#    assert np.allclose(X.mean(axis=0), np.zeros(m))
    # Compute covariance matrix
    C = np.dot(X.T, X) / (n-1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    return eigen_vecs, eigen_vals

def shadowIndex(R,G,B,NIR):
    """
    Calculates the shadow index, basically the first component of PCA,
    following Hui et al. 2013
    """
    OK = R!=0 # boolean array with data
    
    Red = np.float64(R) 
    r = np.mean(Red[OK]) 
    Green = np.float64(G)
    g = np.mean(Green[OK])
    Blue = np.float64(B) 
    b = np.mean(Blue[OK]) 
    Near = np.float64(NIR)
    n = np.mean(Near[OK])
    del R,G,B,NIR
    
    Red = Red-r 
    Green = Green-g 
    Blue = Blue-b 
    Near = Near-n
    del r, g, b, n
    
    # sampleset
    Nsamp = int(min(1e4,np.sum(OK))) # try to have a sampleset of 10'000 points
    sampSet,other =  np.where(OK) # OK.nonzero()
    sampling = random.choices(sampSet, k=Nsamp)
    del sampSet, Nsamp
    
    Red = Red.ravel(order='F')
    Green = Green.ravel(order='F')
    Blue = Blue.ravel(order='F')
    Near = Near.ravel(order='F')
    
    X = np.transpose(np.array([Red[sampling], Green[sampling], Blue[sampling], Near[sampling]]))
    e,lamb = pca(X) 
    del X
    PC1 = np.dot(e[0,:], np.array([Red, Green, Blue, Near]))
    SI = np.transpose(np.reshape(PC1, OK.shape))
    return SI

def shadeIndex(R,G,B,NIR):
    """
    Amplify dark regions by simple multiplication
    following Altena 2008
    """
    NanBol = R==0 # boolean array with no data
    
    Red = np.float64(R) # /2**16
    Red[~NanBol] = np.interp(Red[~NanBol], (Red[~NanBol].min(), Red[~NanBol].max()), (0, +1))
    Red[NanBol] = 0
    Green = np.float64(G) # /2**16
    Green[~NanBol] = np.interp(Green[~NanBol], (Green[~NanBol].min(), Green[~NanBol].max()), (0, +1))
    Green[NanBol] = 0
    Blue = np.float64(B) # /2**16
    Blue[~NanBol] = np.interp(Blue[~NanBol], (Blue[~NanBol].min(), Blue[~NanBol].max()), (0, +1))
    Blue[NanBol] = 0
    Near = np.float64(NIR) # /2**16 
    Near[~NanBol] = np.interp(Near[~NanBol], (Near[~NanBol].min(), Near[~NanBol].max()), (0, +1))
    Near[NanBol] = 0
    del R,G,B,NIR
    SI = np.prod(np.dstack([(1-Red),(1-Green),(1-Blue),(1-Near)]), axis=2)
    return SI

def ruffenacht(R,G,B,NIR):
    """
    Transform Red Green Blue NIR to shadow intensities/probabilities
    """
    ae = 1e+1 
    be = 5e-1
    
    NanBol = R==0 # boolean array with no data
    
    Red = np.float64(R) # /2**16
    Red[~NanBol] = np.interp(Red[~NanBol], (Red[~NanBol].min(), Red[~NanBol].max()), (0, +1))
    Red[NanBol] = 0
    Green = np.float64(G) # /2**16
    Green[~NanBol] = np.interp(Green[~NanBol], (Green[~NanBol].min(), Green[~NanBol].max()), (0, +1))
    Green[NanBol] = 0
    Blue = np.float64(B) # /2**16
    Blue[~NanBol] = np.interp(Blue[~NanBol], (Blue[~NanBol].min(), Blue[~NanBol].max()), (0, +1))
    Blue[NanBol] = 0
    Near = np.float64(NIR) # /2**16 
    Near[~NanBol] = np.interp(Near[~NanBol], (Near[~NanBol].min(), Near[~NanBol].max()), (0, +1))
    Near[NanBol] = 0
    del R,G,B,NIR
    
    Fk = np.amax(np.stack((Red,Green,Blue),axis=2),axis=2)
    F = np.divide(np.clip(Fk, 0,2) ,2) # (10), see Fedembach 2010
    L = np.divide(Red+Green+Blue, 3) # (4), see Fedembach 2010
    del Red,Green,Blue,Fk
    
    Dvis = S_curve(1-L,ae,be)
    Dnir = S_curve(1-Near,ae,be) # (5), see Fedembach 2010
    D = np.multiply(Dvis,Dnir) # (6), see Fedembach 2010
    del L,Near,Dvis,Dnir
    
    M = np.multiply(D,(1-F)) 
    return M
    
def S_curve(x,a,b):
    fe = -a*(x-b)
    fx = np.divide(1,1+np.exp(fe))
    del fe,x
    return fx

# geometric functions
def getShadowPolygon(M,sizPix,thres):
    mn = np.ceil(np.divide(np.nanprod(M.shape),sizPix));
    SupPix = segmentation.slic(M, sigma = 1, 
                  n_segments=mn, 
                  compactness=0.010) # create super pixels
    
#    g = graph.rag_mean_color(M, SupPix) # create region adjacency graph
#    mc = np.empty(len(g))
#    for n in g:
#        mc[n] = g.nodes[n]['mean color'][1]
#    graphCut = graph.cut_threshold(SupPix, g, thres)
#    meanIm = color.label2rgb(graphCut, M, kind='avg')
    meanIm = color.label2rgb(SupPix, M, kind='avg')
    sturge = 1.6*(math.log2(mn)+1)
    values, base = np.histogram(np.reshape(meanIm,-1), 
                                bins=np.int(np.ceil(sturge)))
    dips = findValley(values,base,2)
    val = max(dips)
#    val = filters.threshold_otsu(meanIm)
#    val = filters.threshold_yen(meanIm)
    imSeparation = meanIm > val
    labels = measure.label(imSeparation, background=0) 
    labels = np.int16(labels) # so it can be used for the boundaries extraction
    return labels, SupPix

def getShadows(M,siz,loop):
    mn = M.size;
    Mmed = M
    for i in range(loop):
        Mmed = ndimage.median_filter(M, size=siz)
    sturge = 1.6*(math.log2(mn)+1)
    values, base = np.histogram(np.reshape(Mmed,-1), 
                                bins=np.int(np.ceil(sturge)))    
    dips = findValley(values,base,2)
    val = max(dips)
    imSeparation = Mmed > val
    labels = measure.label(imSeparation, background=0) 
    return labels

def findValley(values,base,neighbors):
    """
    A valley is a point which has "n" consecuative higher values on both sides
    """
    for i in range(neighbors):
        if i == 0:
            wallP = np.roll(values, +(i+1))
            wallM = np.roll(values, -(i+1))
        else:
            wallP = np.vstack((wallP, np.roll(values, +(i+1))))
            wallM = np.vstack((wallM, np.roll(values, -(i+1))))            
    if neighbors>1:
        concavP = np.all(np.sign(np.diff(wallP, n=1, axis=0))==+1, axis=0)
        concavM = np.all(np.sign(np.diff(wallM, n=1, axis=0))==+1, axis=0)
    selec = np.all(np.vstack((concavP,concavM)),axis=0)    
    selec = selec[neighbors-1:-(neighbors+1)]
    idx = base[neighbors-1:-(neighbors+2)]
    dips = idx[selec]
    #walls = np.amin(sandwich,axis=0)
    #selec = values<walls    
    return dips
    
def castOrientation(I,sunZn,sunAz):
#    kernel = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    kernel = np.array([[17], [61], [17]])*np.array([-1, 0, 1])/95
    Idx = ndimage.convolve(I, kernel) # steerable filters
    Idy = ndimage.convolve(I, np.flip(np.transpose(kernel),axis=0))
    Ican = np.multiply(np.cos(np.radians(sunAz)),Idy) - np.multiply(np.sin(np.radians(sunAz)),Idx)
    return Ican

def bboxBoolean(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def RefTrans(Transform,dI,dJ):
    newTransform = (Transform[0]+dI*Transform[1]+dJ*Transform[2], 
                    Transform[1], Transform[2], 
                    Transform[3]+dI*Transform[4]+dJ*Transform[5], 
                    Transform[4], Transform[5])
    return newTransform

def RefScale(Transform,scaling):
    # not using center of pixel
    newTransform = (Transform[0], Transform[1]*scaling, Transform[2]*scaling,                    
                    Transform[3], Transform[4]*scaling, Transform[5]*scaling)    
    return newTransform

def rotMat(theta):
    """
    build rotation matrix, theta is in degrees
    """
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))], 
                  [np.sin(np.radians(theta)), np.cos(np.radians(theta))]]);
    return R

def labelOccluderAndCasted(labels, sunZn, sunAz, subTransform):
    labels = labels.astype(np.int32)
    msk = labels>=1
    mL,nL = labels.shape
    shadowIdx = np.zeros((mL,nL), dtype=np.int16)
    inner = ndimage.morphology.binary_erosion(msk)
    bndOrient = castOrientation(inner.astype(np.float),sunZn,sunAz)
    del mL,nL,inner
    
    for i in range(1,labels.max()): # loop through all polygons
        loc = ndimage.find_objects(labels==i, max_label=2)[0]
        if loc is not None:
            subLabel = labels[loc] # generate subset covering only the polygon 
            subLabel[subLabel!=i] = 0   
            subOrient = np.sign(bndOrient[loc])    
            
            subMsk = subLabel==i
            subBound = subMsk ^ ndimage.morphology.binary_erosion(subMsk)
            subOrient[~subBound] = 0
            
            subAz = sunAz[loc]
            
            subWhe = np.nonzero(subMsk)
            ridgIdx = subOrient[subWhe[0],subWhe[1]]==1
            try:
                ridgeI = subWhe[0][ridgIdx]
                ridgeJ = subWhe[1][ridgIdx]
            except IndexError:
                continue    
            
            cast = subOrient==-1 # boundary of the polygon that receives cast shadow
            
            m,n = subMsk.shape
            subShadowIdx = np.zeros((m,n), dtype=np.int16)
            for x in range(len(ridgeI)): # loop through all occluders
                sunDir = subAz[ridgeI[x]][ridgeJ[x]] # degrees [-180 180]
                
                # Bresenham's line algorithm
                dI = -math.cos(math.radians(subAz[ridgeI[x]][ridgeJ[x]])) # flip axis to get from world into image coords
                dJ = -math.sin(math.radians(subAz[ridgeI[x]][ridgeJ[x]])) #
                if abs(sunDir)>90: # northern or southern hemisphere
                    if dI>dJ:
                        rr = np.arange(start=0, stop=ridgeI[x], step=1)
                        cc = np.round(rr*dJ) + ridgeJ[x]
                    else:
                        cc = np.arange(start=0, stop=ridgeJ[x], step=1)
                        rr = np.round(rr*dI) + ridgeI[x]  
                else:
                    if dI>dJ:
                        rr = np.arange(start=ridgeI[x], stop=m, step=1)
                        cc = np.round(rr*dJ) + ridgeJ[x]
                    else:
                        cc = np.arange(start=ridgeJ[x], stop=n, step=1)
                        rr = np.round(rr*dI) + ridgeI[x]   
                # generate cast line in sub-image        
                rr = rr.astype(np.int64)
                cc = cc.astype(np.int64)
                subCast = np.zeros((m, n), dtype=np.uint8)
                
                IN = (cc>0) & (cc<m) & (rr>0) & (rr<n) # inside sub-image                
                if IN.any():
                    rr = rr[IN]
                    cc = cc[IN]
                    try:
                        subCast[rr, cc] = 1
                    except IndexError:
                        continue                
                    
                    # find closest casted
                    castHit = cast&subCast
                    castIdx = castHit[subWhe[0],subWhe[1]]==True
                    castI = subWhe[0][castIdx]
                    castJ = subWhe[1][castIdx] 
                    del IN, castIdx, castHit, subCast, rr, cc, dI, dJ, sunDir
                    
                    if len(castI)>1:
                        # do selection, of the closest casted
                        dist = np.sqrt((castI-ridgeI[x])**2 + (castJ-ridgeJ[x])**2)
                        idx = np.where(dist == np.amin(dist))
                        castI = castI[idx[0]]
                        castJ = castJ[idx[0]]   
                    
                    if len(castI)>0:
                        # write out
                        subShadowIdx[ridgeI[x]][ridgeJ[x]] = x # ridge
                        subShadowIdx[castI[0]][castJ[0]] = x # casted
            shadowIdx[loc] = subShadowIdx        
    return shadowIdx
    
def listOccluderAndCasted(labels, sunZn, sunAz, geoTransform):
    msk = labels>1
    labels = labels.astype(np.int32)
    mskOrient = castOrientation(msk.astype(np.float),sunZn,sunAz)
    mskOrient = np.sign(mskOrient)
    #makeGeoIm(mskOrient,subTransform,crs,"polyRidges.tif")
    
    castList = []
    for shp, val in shapes(labels, mask=msk, connectivity=8):
    #        coord = shp["coordinates"]
    #        coord = np.uint16(np.squeeze(np.array(coord[:])))    
        if val!=0:
    #    if val==48:    
            # get ridge coordinates
            polygoon = shape(shp)
            polyRast = labels==val # select the polygon
            polyInnr = ndimage.binary_erosion(polyRast, np.ones((3,3), dtype=bool))
            polyBoun = np.logical_xor(polyRast, polyInnr)
            polyWhe = np.nonzero(polyBoun)
            ridgIdx = mskOrient[polyWhe[0],polyWhe[1]]==1
            ridgeI = polyWhe[0][ridgIdx]
            ridgeJ = polyWhe[1][ridgIdx]    
            del polyRast, polyInnr, polyBoun, polyWhe, ridgIdx
        
            for x in range(len(ridgeI)): #ridgeI:
                try:
                    castLine = LineString([[ridgeJ[x],ridgeI[x]],
                                [ridgeJ[x] - (math.sin(math.radians(sunAz[ridgeI[x]][ridgeJ[x]]))*1e4), 
                                 ridgeI[x] + (math.cos(math.radians(sunAz[ridgeI[x]][ridgeJ[x]]))*1e4)]])
                except IndexError:
                    continue
                try:
                    castEnd = polygoon.intersection(castLine)
                except TopologicalError:
                    # somehow the exterior of the polygon crosses or touches itself, making it a LinearRing
                    polygoon = polygoon.buffer(0) 
                    castEnd = polygoon.intersection(castLine)

                if castEnd.geom_type == 'LineString':
                    castEnd = castEnd.coords[:]
                elif castEnd.geom_type == 'MultiLineString':
                    # castEnd = [list(x.coords) for x in list(castEnd)]
                    cEnd = []
                    for m in list(castEnd):
                        cEnd = cEnd + m.coords[:]
                    castEnd = cEnd
                    del m, cEnd
                elif castEnd.geom_type == 'GeometryCollection':
                    cEnd = []
                    for m in range(len(castEnd)):
                        cEnd = cEnd + castEnd[m].coords[:]
                    castEnd = cEnd
                    del m, cEnd
                elif castEnd.geom_type == 'Point':
                    castEnd = []
                else:
                    print('something went wrong?')
                
                # if empty
                if len(castEnd)>1:
                # if len(castEnd.coords[:])>1:
                    # find closest intersection
                    occluder = Point(ridgeJ[x],ridgeI[x])
                    #dists = [Point(c).distance(occluder) for c in castEnd.coords]
                    dists = [Point(c).distance(occluder) for c in castEnd]
                    dists = [float('Inf') if  i == 0 else i for i in dists]
                    castIdx = dists.index(min(dists)) 
                    casted = castEnd[castIdx]
                    
                    # transform to UTM and append to array
                    ridgeX = ridgeI[x]*geoTransform[2] + ridgeJ[x]*geoTransform[1] + geoTransform[0]
                    ridgeY = ridgeI[x]*geoTransform[5] + ridgeJ[x]*geoTransform[4] + geoTransform[3]
                    castX = casted[1]*geoTransform[2] + casted[0]*geoTransform[1] + geoTransform[0]
                    castY = casted[1]*geoTransform[5] + casted[0]*geoTransform[4] + geoTransform[3]
                    
                    
                    castLine = np.array([ridgeX, ridgeY, castX, castY, 
                                         sunAz[ridgeI[x]][ridgeJ[x]] , 
                                         sunZn[ridgeI[x]][ridgeJ[x]] ])
                    castList.append(castLine)
                    del dists, occluder, castIdx, casted
                del castLine, castEnd
    return castList

# image matching functions
def LucasKanade(I1, I2, window_size, stepSize=False, tau=1e-2): 
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])*.25
    radius = np.floor(window_size/2).astype('int') # window_size should be odd
    
    fx = ndimage.convolve(I1, kernel_x)
    fy = ndimage.convolve(I1, np.flip(np.transpose(kernel_x),axis=0))
    ft = ndimage.convolve(I2, kernel_t) + ndimage.convolve(I1, -kernel_t)
    
    # double loop to visit all pixels 
    if stepSize:
        stepsI = np.arange(radius, I1.shape[0]-radius, window_size)
        stepsJ = np.arange(radius, I1.shape[1]-radius, window_size)
        u = np.zeros((len(stepsI),len(stepsJ)))
        v = np.zeros((len(stepsI),len(stepsJ)))
    else:
        stepsI = range(radius, I1.shape[0]-radius)
        stepsJ = range(radius, I1.shape[1]-radius)
        u = np.zeros(I1.shape)
        v = np.zeros(I1.shape)
    
    for iIdx in range(len(stepsI)):
        i = stepsI[iIdx]
        for jIdx in range(len(stepsJ)):
            j = stepsI[jIdx]
            # get templates
            Ix = fx[i-radius:i+radius+1, j-radius:j+radius+1].flatten()
            Iy = fy[i-radius:i+radius+1, j-radius:j+radius+1].flatten()
            It = ft[i-radius:i+radius+1, j-radius:j+radius+1].flatten()
            
            # look if variation is present
            if np.std(It)!=0:
                b = np.reshape(It, (It.shape[0],1)) # get b here
                A = np.vstack((Ix, Iy)).T # get A here
                # threshold tau should be larger than the smallest eigenvalue of A'A
                if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A))))>=tau:
                    nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
                    u[iIdx,jIdx]=nu[0]
                    v[iIdx,jIdx]=nu[1]
 
    return (u,v)

def getNetworkIndices(n):
    '''
    for a number (n) of images, generate a list with all matchable combinations
    '''
    Grids = np.indices((len(s2Path),len(s2Path)))
    Grid1 = np.triu(Grids[0]+1,+1).flatten()
    Grid1 = Grid1[Grid1 != 0]-1
    Grid2 = np.triu(Grids[1]+1,+1).flatten()
    Grid2 = Grid2[Grid2 != 0]-1
    GridIdxs = np.vstack((Grid1, Grid2))
    return GridIdxs

# I/O functions
def makeGeoIm(I,R,crs,fName):
    drv = gdal.GetDriverByName("GTiff") # export image
    # type
    if I.dtype=='float64':
        ds = drv.Create(fName, I.shape[1], I.shape[0], 6, gdal.GDT_Float64)
    else:
        ds = drv.Create(fName, I.shape[1], I.shape[0], 6, gdal.GDT_Int32)
    ds.SetGeoTransform(R)
    ds.SetProjection(crs)
    ds.GetRasterBand(1).WriteArray(I)
    ds = None
    del ds    

datPath = '/Users/Alten005/surfdrive/Eratosthenes/Denali/' 
#s2Path = 'Data/S2A_MSIL1C_20180225T214531_N0206_R129_T05VPL_20180225T232042/'
#fName = 'T05VPL_20180225T214531_B'

s2Path = ('Data/S2A_MSIL1C_20180225T214531_N0206_R129_T05VNK_20180225T232042/',
          'Data/S2B_MSIL1C_20190225T214529_N0207_R129_T05VNK_20190226T010218/',
          'Data/S2A_MSIL1C_20200225T214531_N0209_R129_T05VNK_20200225T231600/')
fName = ('T05VNK_20180225T214531_B', 'T05VNK_20190225T214529_B', 'T05VNK_20200225T214531_B')

# do a subset of the imagery
minI = 4000
maxI = 6000

for i in range(len(s2Path)):
    print(i)
    sen2Path = datPath + s2Path[i]
    
    # read imagery of the different bands
    (B2, crs, geoTransform, targetprj) = read_band_image('02', sen2Path)
    (B3, crs, geoTransform, targetprj) = read_band_image('03', sen2Path)
    (B4, crs, geoTransform, targetprj) = read_band_image('04', sen2Path)
    (B8, crs, geoTransform, targetprj) = read_band_image('08', sen2Path)
    
    mI = np.size(B2,axis=0)
    nI = np.size(B2,axis=1)
    
    # reduce image space, so it fit in memory
    B2 = B2[minI:maxI,minI:maxI]
    B3 = B3[minI:maxI,minI:maxI]
    B4 = B4[minI:maxI,minI:maxI]
    B8 = B8[minI:maxI,minI:maxI]
    
    subTransform = RefTrans(geoTransform,minI,minI) # create georeference for subframe
        
    # transform to shadow image
    M = ruffenacht(B2,B3,B4,B8)
    makeGeoIm(M,subTransform,crs,sen2Path + "ruffenacht.tif")
    # M = shadowIndex(B2,B3,B4,B8)
    # makeGeoIm(M,subTransform,crs,sen2Path + "swIdx.tif") # pc1
    # M = shadeIndex(B2,B3,B4,B8)
    # makeGeoIm(M,subTransform,crs,sen2Path + "shIdx.tif")
    # M = nsvi(B2,B3,B4)
    # makeGeoIm(M,subTransform,crs,sen2Path + "nsvi.tif")
    # M = mpsi(B2,B3,B4)
    # makeGeoIm(M,subTransform,crs,sen2Path + "mpsi.tif")
    
    del B2,B3,B4,B8
    
    if not os.path.exists(sen2Path + 'labelCastConn.tif'):
        # classify into regions
        siz = 5
        loop = 500
        labels = getShadows(M,siz,loop)
        makeGeoIm(labels,subTransform,crs, sen2Path + "schur.tif")
        
        # find self-shadow and cast-shadow
        (sunZn,sunAz) = read_sun_angles(sen2Path)
        sunZn = sunZn[minI:maxI,minI:maxI]
        sunAz = sunAz[minI:maxI,minI:maxI]
        
        # raster-based
        castList = labelOccluderAndCasted(labels, sunZn, sunAz, subTransform)
        # write data 
        
        makeGeoIm(labels,subTransform,crs,sen2Path + "labelPolygons.tif")
        makeGeoIm(castList,subTransform,crs,sen2Path + "labelCastConn.tif")
        
#        # polygon-based
#        castList = listOccluderAndCasted(labels, sunZn, sunAz, subTransform)
#        # write data to txt-file
#        # Header = ('ridgeX', 'ridgeY', 'castX', 'castY', 'sunAzi', 'sunZn')
#        fCast = sen2Path+fName[i][0:-2]+'.txt'
#        with open(fCast, 'w') as f:
#            for item in castList:
#                np.savetxt(f,item.reshape(1,6), fmt="%.2f", delimiter=",", 
#                           newline='\n', header='', footer='', comments='# ')
#        #        np.savetxt("test2.txt", x, fmt="%2.1f", delimiter=",")
#        #        f.write("%s\n" % item)
        print('wrote '+ fName[i][0:-2])

## co-register
# get shadow images into one array
for i in range(len(s2Path)):
    sen2Path = datPath + s2Path[i]
    # get sun orientation
    (sunZn,sunAz) = read_sun_angles(sen2Path)
    sunZn = sunZn[minI:maxI,minI:maxI]
    sunAz = sunAz[minI:maxI,minI:maxI]
    # read shadow image
    img = gdal.Open(sen2Path + "ruffenacht.tif")
    M = np.array(img.GetRasterBand(1).ReadAsArray())   
    # create ridge image 
    Mcan = castOrientation(M,sunZn,sunAz) 
    # Mcan = castOrientation(msk,sunZn,sunAz) 
    Mcan[Mcan<0] = 0
    # stack into 3D array
    if i==0:
        Mstack = Mcan
    elif i==1:
        Mstack = np.stack((Mstack, Mcan), axis=2)
    else:
        Mstack = np.dstack((Mstack, Mcan))    

# get network
GridIdxs = getNetworkIndices(len(s2Path))
tempSize = 15
stepSize = True
# get observation angle
(obsZn,obsAz) = read_view_angles(sen2Path)
obsZn = obsZn[minI:maxI,minI:maxI]
obsAz = obsAz[minI:maxI,minI:maxI]
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
for i in range(GridIdxs.shape[1]):
    # optical flow following Lukas Kanade
    
    #####################
    # blur with gaussian?
    #####################
    
    (y_N,y_E) = LucasKanade(Mstack[:,:,GridIdxs[0,i]], Mstack[:,:,GridIdxs[1,i]],
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
    
    # build 2*2*n covariance matrix
    q_yy = np.zeros((len(y_Zn),2,2), dtype = float)
    for b in range(len(y_Zn)):
        q_yy[b,:,:] = np.matmul(rotMat(y_Az[b]),np.array([[sig_y, 0], [0, (1+np.tan(np.radians(y_Zn[b])))*sig_y]]))
    # transform to diagonal matrix
    Q_yy = block_diag(*q_yy)  
    del q_yy,y_N,y_E,y_Zn,y_Az
    
    x_hat = np.linalg.lstsq(A, y, rcond=None)
    gls_model = sm.GLS(A, y, sigma=Q_yy)
    x_hat = gls_model.fit()

    
    if i==0:
        Dstack = D
    else:
        Dstack = Dstack + D

#lkTransform = RefScale(RefTrans(subTransform,tempSize/2,tempSize/2),tempSize)   
#makeGeoIm(Dstack[0],lkTransform,crs,"DispAx1.tif")

      

    

        






# weights through observation angle

    
# makeGeoIm(Mcan,subTransform,crs,"shadowRidges.tif")

# merge lists
for i in range(len(s2Path)):
    print('getting together')
    
#processed_image = cv2.filter2D(image,-1,kernel)

(viwZn,viwAz) = read_view_angles(sen2Path)
viwZn = viwZn[minI:maxI,minI:maxI]
viwAz = viwAz[minI:maxI,minI:maxI]



fig, ax = plt.subplots()
im = ax.imshow(viwZn)
fig.colorbar(im)
plt.show()





