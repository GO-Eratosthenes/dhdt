import glob
import os
from xml.etree import ElementTree

import numpy as np

import geopandas

def get_bbox_from_tile_code(tile_code, \
                            shp_dir='/Users/Alten005/surfdrive/Eratosthenes/SatelliteTiles', \
                            shp_name='sentinel2_tiles_world.shp'):
    tile_code = tile_code.upper()
    shp_path = os.path.join(shp_dir,shp_name)
    
    mgrs = geopandas.read_file(shp_path)
    
    toi = mgrs[mgrs['Name']==tile_code]
    return toi

def get_array_from_xml(treeStruc):  # generic
    """
    Arrays within a xml structure are given line per line
    Output is an array
    """
    for i in range(0, len(treeStruc)):
        Trow = [float(s) for s in treeStruc[i].text.split(' ')]
        if i == 0:
            Tn = Trow
        elif i == 1:
            Trow = np.stack((Trow, Trow), 0)
            Tn = np.stack((Tn, Trow[1, :]), 0)
        else:
            # Trow = np.stack((Trow, Trow),0)
            # Tn = np.concatenate((Tn, [Trow[1,:]]),0)
            Tn = np.concatenate((Tn, [Trow]), 0)
    return Tn

def get_S2_image_locations(fname):
    """
    The Sentinel-2 imagery are placed within a folder structure, where one 
    folder has an ever changing name, when this function finds the path from
    the meta data

    Parameters
    ----------
    fname : string
        path string to the Sentinel-2 folder

    Returns
    -------
    im_paths : list, size=(13,)
        lists with relative folder and file locations of the bands
    datastrip_id : str
        folder name of the metadata
        
    Example
    -------
    >>> import os
    >>> fpath = '/Users/Data/'
    >>> sname = 'S2A_MSIL1C_20200923T163311_N0209_R140_T15MXV_20200923T200821.SAFE'
    >>> fname = 'MTD_MSIL1C.xml'
    >>> full_path = os.path.join(fpath, sname, fname)
    >>> im_paths,datastrip_id = get_S2_image_locations(full_path)
    >>> im_paths
    ['GRANULE/L1C_T15MXV_A027450_20200923T163313/IMG_DATA/T15MXV_20200923T163311_B01', 
     'GRANULE/L1C_T15MXV_A027450_20200923T163313/IMG_DATA/T15MXV_20200923T163311_B02']   
    >>> datastrip_id
    'S2A_OPER_MSI_L1C_DS_VGS1_20200923T200821_S20200923T163313_N02.09'
    """
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()
    granule_list = root[0][0][11][0][0]
    datastrip_id = granule_list.get('datastripIdentifier')
    
    im_paths = []
    for im_loc in root.iter('IMAGE_FILE'):
        im_paths.append(im_loc.text)
    #    print(im_loc.text)
    
    return im_paths, datastrip_id

def meta_S2string(S2str):
    """ get meta information of the Sentinel-2 file name

    Parameters
    ----------
    S2str : string
        filename of the L1C data

    Returns
    -------
    S2time : string
        date "+YYYY-MM-DD"
    S2orbit : string
        relative orbit "RXXX"
    S2tile : string
        tile code "TXXXXX"

    Example
    -------
    >>> S2str = 'S2A_MSIL1C_20200923T163311_N0209_R140_T15MXV_20200923T200821.SAFE'
    >>> S2time, S2orbit, S2tile = meta_S2string(S2str)
    >>> S2time
    '+2020-09-23'
    >>> S2orbit
    'R140'
    >>> S2tile
    'T15MXV'
    """
    assert type(S2str)==str, ("please provide a string")
    assert S2str[0:2]=='S2', ("please provide a Sentinel-2 file string")
    S2split = S2str.split('_')
    S2time = S2split[2][0:8]
    # convert to +YYYY-MM-DD string 
    # then it can be in the meta-data of following products
    S2time = '+' + S2time[0:4] +'-'+ S2time[4:6] +'-'+ S2time[6:8]
    S2orbit = S2split[4]
    S2tile = S2split[5]
    return S2time, S2orbit, S2tile

def get_S2_folders(im_path):
    s2_list = [x for x in os.listdir(im_path) 
               if (os.path.isdir(os.path.join(im_path,x))) & (x[0:2]=='S2')]
    return s2_list

def get_tiles_from_S2_list(S2_list):
    tile_list = [x.split('_')[5] for x in S2_list]
    tile_list = list(set(tile_list))
    return tile_list

def get_utm_from_S2_tiles(tile_list):
    utm_list = [int(x[1:3]) for x in tile_list]
    utm_list = list(set(utm_list))
    return utm_list

