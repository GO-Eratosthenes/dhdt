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
    with open(fname, 'r') as f:
        metadata_text = f.read()
    Zn, Az = extract_sun_angles_grid_s2(metadata_text)
    return Zn, Az


def extract_sun_angles_grid_s2(metadata_text):
    """
    This function parses the content of the xml-file of the Sentinel-2 scene
    and extracts an array with sun angles, as these vary along the scene.
    input:   metadata_text  string            content of the xml-file of
                                              Sentinel-2
    output:  Zn             array (n x m)     array of the Zenith angles
             Az             array (n x m)     array of the Azimuth angles

    """
    root = ElementTree.fromstring(metadata_text)

    # image dimensions
    for meta in root.iter('Size'):
        res = float(meta.get('resolution'))
        if res == 10:  # take 10 meter band
            mI = float(meta[0].text)
            nI = float(meta[1].text)

    # get Zenith array
    Zenith = root[1][1][0][0][2]
    Zn = get_array_from_xml(Zenith)

    zi = np.linspace(0 - 10, mI + 10, np.size(Zn, axis=0))
    zj = np.linspace(0 - 10, nI + 10, np.size(Zn, axis=1))
    Zi, Zj = np.meshgrid(zi, zj)
    Zij = np.dstack([Zi, Zj]).reshape(-1, 2)

    iGrd = np.arange(0, mI)
    jGrd = np.arange(0, nI)
    Igrd, Jgrd = np.meshgrid(iGrd, jGrd)

    Zn = griddata(Zij, Zn.reshape(-1), (Igrd, Jgrd), method="linear")

    # get Azimuth array
    Azimuth = root[1][1][0][1][2]
    Az = get_array_from_xml(Azimuth)

    ai = np.linspace(0 - 10, mI + 10, np.size(Az, axis=0))
    aj = np.linspace(0 - 10, nI + 10, np.size(Az, axis=1))
    Ai, Aj = np.meshgrid(ai, aj)
    Aij = np.dstack([Ai, Aj]).reshape(-1, 2)

    Az = griddata(Aij, Az.reshape(-1), (Igrd, Jgrd), method="linear")
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
