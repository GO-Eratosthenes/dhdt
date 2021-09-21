# generic libraries
import glob
import os

from xml.etree import ElementTree

import numpy as np

# geospatial libaries
from osgeo import gdal, osr

# raster/image libraries
from PIL import Image, ImageDraw
from scipy.interpolate import griddata

from ..generic.handler_s2 import get_array_from_xml

from eratosthenes.generic.mapping_tools import map2pix

def s2_dn2toa(I):
    """convert the digital numbers of Sentinel-2 to top of atmosphere (TOA)
    
    Notes
    -----
    sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/
    level-1c/algorithm
    """
    no_data = 0
    
    I = I*10000
    return I       

def read_band_s2(band, path):  # pre-processing
    """
    This function takes as input the Sentinel-2 band name and the path of the
    folder that the images are stored, reads the image and returns the data as
    an array
    input:   band           string            Sentinel-2 band name
             path           string            path of the folder
    output:  data           array (n x m)     array of the band image
             spatialRef     string            projection
             geoTransform   tuple             affine transformation
                                              coefficients
             targetprj                        spatial reference
    """
    fname = os.path.join(path, '*B'+band+'.jp2')
    img = gdal.Open(glob.glob(fname)[0])
    data = np.array(img.GetRasterBand(1).ReadAsArray())
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt=img.GetProjection())
    return data, spatialRef, geoTransform, targetprj

def read_sun_angles_s2(path):  # pre-processing
    """
    This function reads the xml-file of the Sentinel-2 scene and extracts an
    array with sun angles, as these vary along the scene.
    input:   path           string            path where xml-file of
                                              Sentinel-2 is situated
    output:  Zn             array (n x m)     array of the Zenith angles
             Az             array (n x m)     array of the Azimtuh angles

    """
    fname = os.path.join(path, 'MTD_TL.xml')
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()

    # image dimensions
    for meta in root.iter('Size'):
        res = float(meta.get('resolution'))
        if res == 10:  # take 10 meter band
            mI = float(meta[0].text)
            nI = float(meta[1].text)

    # get Zenith array
    Zenith = root[1][1][0][0][2]
    Zn = get_array_from_xml(Zenith)
    znSpac = float(root[1][1][0][0][0].text)
    znSpac = np.stack((znSpac, float(root[1][1][0][0][1].text)), 0)
    znSpac = np.divide(znSpac, 10)  # transform from meters to pixels

    zi = np.linspace(0 - 10, mI + 10, np.size(Zn, axis=0))
    zj = np.linspace(0 - 10, nI + 10, np.size(Zn, axis=1))
    Zi, Zj = np.meshgrid(zi, zj)
    Zij = np.dstack([Zi, Zj]).reshape(-1, 2)
    del zi, zj, Zi, Zj, znSpac

    iGrd = np.arange(0, mI)
    jGrd = np.arange(0, nI)
    Igrd, Jgrd = np.meshgrid(iGrd, jGrd)

    Zn = griddata(Zij, Zn.reshape(-1), (Igrd, Jgrd), method="linear")

    # get Azimuth array
    Azimuth = root[1][1][0][1][2]
    Az = get_array_from_xml(Azimuth)
    azSpac = float(root[1][1][0][1][0].text)
    azSpac = np.stack((azSpac, float(root[1][1][0][1][1].text)), 0)

    ai = np.linspace(0 - 10, mI + 10, np.size(Az, axis=0))
    aj = np.linspace(0 - 10, nI + 10, np.size(Az, axis=1))
    Ai, Aj = np.meshgrid(ai, aj)
    Aij = np.dstack([Ai, Aj]).reshape(-1, 2)
    del ai, aj, Ai, Aj, azSpac

    Az = griddata(Aij, Az.reshape(-1), (Igrd, Jgrd), method="linear")
    del Igrd, Jgrd, Zij, Aij
    return Zn, Az

def read_detector_mask(path_meta, msk_dim, boi, geoTransform):   
    det_stack = np.zeros(msk_dim, dtype='int8')    
    for i in range(len(boi)):
        im_id = boi[i]
        f_meta = os.path.join(path_meta, 'MSK_DETFOO_B'+ f'{im_id:02.0f}' + '.gml')
        dom = ElementTree.parse(glob.glob(f_meta)[0])
        root = dom.getroot()  
        
        mask_members = root[2]
        for k in range(len(mask_members)):
            # get detector number from meta-data
            det_id = mask_members[k].attrib
            det_id = list(det_id.items())[0][1].split('-')[2]
            det_num = int(det_id)
        
            # get footprint
            pos_dim = mask_members[k][1][0][0][0][0].attrib
            pos_dim = int(list(pos_dim.items())[0][1])
            pos_list = mask_members[k][1][0][0][0][0].text
            pos_row = [float(s) for s in pos_list.split(' ')]
            pos_arr = np.array(pos_row).reshape((int(len(pos_row)/pos_dim), pos_dim))
            
            # transform to image coordinates
            i_arr, j_arr = map2pix(geoTransform, pos_arr[:,0], pos_arr[:,1])
            ij_arr = np.hstack((j_arr[:,np.newaxis], i_arr[:,np.newaxis]))
            # make mask
            msk = Image.new("L", [np.size(det_stack,0), np.size(det_stack,1)], 0)
            ImageDraw.Draw(msk).polygon(tuple(map(tuple, ij_arr[:,0:2])), \
                                        outline=det_num, fill=det_num)
            msk = np.array(msk)    
            det_stack[:,:,i] = np.maximum(det_stack[:,:,i], msk)
    return det_stack

def read_cloud_mask(path_meta, msk_dim, geoTransform):   
    msk_clouds = np.zeros(msk_dim, dtype='int8')    

    f_meta = os.path.join(path_meta, 'MSK_CLOUDS_B00.gml')
    dom = ElementTree.parse(glob.glob(f_meta)[0])
    root = dom.getroot()  
    
    mask_members = root[2]
    for k in range(len(mask_members)):    
        # get footprint
        pos_dim = mask_members[k][1][0][0][0][0].attrib
        pos_dim = int(list(pos_dim.items())[0][1])
        pos_list = mask_members[k][1][0][0][0][0].text
        pos_row = [float(s) for s in pos_list.split(' ')]
        pos_arr = np.array(pos_row).reshape((int(len(pos_row)/pos_dim), pos_dim))
        
        # transform to image coordinates
        i_arr, j_arr = map2pix(geoTransform, pos_arr[:,0], pos_arr[:,1])
        ij_arr = np.hstack((j_arr[:,np.newaxis], i_arr[:,np.newaxis]))
        # make mask
        msk = Image.new("L", [msk_dim[0], msk_dim[1]], 0)
        ImageDraw.Draw(msk).polygon(tuple(map(tuple, ij_arr[:,0:2])), \
                                    outline=1, fill=1)
        msk = np.array(msk)    
        msk_clouds = np.maximum(msk_clouds, msk)
    return msk_clouds

def read_detector_time(path):
    det_time = np.zeros((13, 12), dtype='datetime64[ns]')
    det_name = [None] * 13
    det_meta = np.zeros((13, 4), dtype='float')
    fname = os.path.join(path, 'MTD_DS.xml')
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()

    # image dimensions
    for meta in root.iter('Band_Time_Stamp'):
        bnd = int(meta.get('bandId')) # 0..12
        # <Spectral_Information bandId="8" physicalBand="B8A">
        for stamps in meta:
            det = int(stamps.get('detectorId'))            
            det_time[bnd,det-1] = np.datetime64(stamps[1].text, 'ns') 
            
    for meta in root.iter('Spectral_Information'):
        bnd = int(meta.get('bandId'))
        bnd_name = meta.get('physicalBand')
        # at a zero if needed, this seems the correct naming convention
        if len(bnd_name)==2: 
            bnd_name = bnd_name[0]+'0'+bnd_name[-1]
        det_name[bnd] = bnd_name
        det_meta[bnd,0] = float(meta[0].text) # resolution [m]
        det_meta[bnd,1] = float(meta[1][0].text) # min
        det_meta[bnd,2] = float(meta[1][1].text) # max
        det_meta[bnd,3] = float(meta[1][2].text) # mean
    return det_time, det_name, det_meta