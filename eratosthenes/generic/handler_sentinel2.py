import os
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
    """
    dom = ElementTree.parse(glob.glob(fname)[0])
    root = dom.getroot()
    granule_list = root[0][0][11][0][0]
    
    im_paths = []
    for im_loc in root.iter('IMAGE_FILE'):
        im_paths.append(im_loc.text)
    #    print(im_loc.text)
    return im_paths

def meta_S2string(S2str):  # generic
    """
    get meta data of the Sentinel-2 file name
    input:   S2str          string            filename of the L1C data
    output:  S2time         string            date "+YYYY-MM-DD"
             S2orbit        string            relative orbit "RXXX"
             S2tile         string            tile code "TXXXXX"
    """
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

