
from ..generic.handler_im import get_image_subset

from eratosthenes.generic.mapping_tools import RefTrans
from eratosthenes.preprocessing.read_s2 import read_band_s2
from eratosthenes.preprocessing.shadow_transforms import enhance_shadow

def create_shadow_image(dat_path, im_name, shadow_transform='reffenacht', \
                        minI=0, maxI=0, minJ=0, maxJ=0):
    """
    Given a specific method, employ shadow transform
    input:   dat_path       string            directory the image is residing
             im_name        string            name of one or the multispectral image     
    output:  M              array (n x m)     shadow enhanced satellite image
    """
    bandList = get_shadow_bands(im_name) # get band numbers of the multispectral images
    (Blue, Green, Red, Nir, crs, geoTransform, targetprj) = read_shadow_bands( \
                dat_path + im_name, bandList)
    if (minI!=0 or maxI!=0 and minI!=0 or maxI!=0):        
        # reduce image space, so it fits in memory
        Blue = get_image_subset(Blue, minI, maxI, minJ, maxJ)
        Green = get_image_subset(Green, minI, maxI, minJ, maxJ)
        Red = get_image_subset(Red, minI, maxI, minJ, maxJ)
        Nir = get_image_subset(Nir, minI, maxI, minJ, maxJ)
            
        geoTransform = RefTrans(geoTransform,minI,minJ) # create georeference for subframe
    else:
        geoTransform = geoTransform
    
    # transform to shadow image  
    M = enhance_shadow(shadow_transform,Blue,Green,Red,Nir)
    return M, geoTransform, crs

def get_shadow_bands(satellite_name):
    """
    Given the name of the satellite, provide the number of the bands that are:
        Blue, Green, Red and Near-infrared
    If the instrument has a high resolution panchromatic band, 
    this is given as the fifth entry    

    input:   satellite_name string            name of the satellite or instrument abreviation
    output:  band_num       list (1 x 4)      list of the band numbers
    """
    
    satellite_name = satellite_name.lower()
    
    # compare if certain segments are present in a string
    
    if len([n for n in ['sentinel','2'] if n in satellite_name])==2 or len([n for n in ['msi'] if n in satellite_name])==1:
        band_num = [2,3,4,8] 
    elif len([n for n in ['landsat','8'] if n in satellite_name])==2 or len([n for n in ['oli'] if n in satellite_name])==1:
        band_num = [2,3,4,5,8] 
    elif len([n for n in ['landsat','7'] if n in satellite_name])==2 or len([n for n in ['etm+'] if n in satellite_name])==1:
        band_num = [1,2,3,4,8]
    elif len([n for n in ['landsat','5'] if n in satellite_name])==2 or len([n for n in ['tm','mss'] if n in satellite_name])==1:
        band_num = [1,2,3,4]
    elif len([n for n in ['rapid','eye'] if n in satellite_name])==2:
        band_num = [1,2,3,5]
    elif len([n for n in ['planet'] if n in satellite_name])==1  or len([n for n in ['dove'] if n in satellite_name])==1:
        band_num = [1,2,3,5]
    elif len([n for n in ['aster'] if n in satellite_name])==1:
        band_num = [1,1,2,3]
    elif len([n for n in ['worldview','3'] if n in satellite_name])==2:        
        band_num = [2,3,5,8]

    return band_num

def read_shadow_bands(sat_path, band_num):
    """
    Reads the specific band numbers of the multispectral satellite images
    input:   sat_path       string            path to imagery and file name
             band_num       list (1 x 4)      list of the band numbers
    output:         
    """
    
    if len([n for n in ['S2','MSIL1C'] if n in sat_path])==2:
        # read imagery of the different bands
        (Blue, crs, geoTransform, targetprj) = read_band_s2( 
            format(band_num[0], '02d'), sat_path)
        (Green, crs, geoTransform, targetprj) = read_band_s2(
            format(band_num[1], '02d'), sat_path)
        (Red, crs, geoTransform, targetprj) = read_band_s2(
            format(band_num[2], '02d'), sat_path)
        (Nir, crs, geoTransform, targetprj) = read_band_s2(
            format(band_num[3], '02d'), sat_path)
    
    # if len(band_num)==5: # include panchromatic band
    #     Pan = ()
    # else:
    #     Pan = ()
    
    return Blue, Green, Red, Nir, crs, geoTransform, targetprj #, Pan