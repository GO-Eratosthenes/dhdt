import numpy as np

from osgeo import gdal, osr

from ..generic.handler_im import get_image_subset

from eratosthenes.generic.mapping_tools import RefTrans, pix2map
from eratosthenes.generic.mapping_io import read_geo_image
from eratosthenes.preprocessing.read_s2 import read_band_s2, read_sun_angles_s2
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

def create_caster_casted_list_from_polygons(dat_path, im_name,bbox=None, 
                                            polygon_id=None):
    """
    Create a list of casted and casted coordinates, of shadow polygons that
    are occluding parts of a glacier
    
    """
    im_path = dat_path + im_name

    (Rgi, spatialRef, geoTransform, targetprj) = read_geo_image(im_path + 'rgi.tif')
    img = gdal.Open(im_path + 'labelPolygons.tif')
    cast = np.array(img.GetRasterBand(1).ReadAsArray())
    img = gdal.Open(im_path + 'labelCastConn.tif')
    conn = np.array(img.GetRasterBand(1).ReadAsArray())
    
    # keep it image-based
    if polygon_id is None:
        selec = Rgi!=0
    else:
        selec = Rgi!=polygon_id
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
    (sunZn,sunAz) = read_sun_angles_s2(im_path)
    #OBS: still in relative coordinates!!!
    if bbox is not None:
        sunZn = get_image_subset(sunZn,bbox)
        sunAz = get_image_subset(sunAz,bbox)
    sunZen = sunZn[castngIJ[:,0],castngIJ[:,1]]
    sunAzi = sunAz[castngIJ[:,0],castngIJ[:,1]]
    
    # write to file
    f = open(dat_path+'conn.txt', 'w')
    for i in range(castedX.shape[0]):
        line = '{:+8.2f}'.format(castngX[i])+' '+'{:+8.2f}'.format(castngY[i])+' '
        line = line + '{:+8.2f}'.format(castedX[i])+' '+'{:+8.2f}'.format(castedY[i])+' '
        line = line + '{:+3.4f}'.format(sunAzi[i])+' '+'{:+3.4f}'.format(sunZen[i])
        f.write(line + '\n')
    f.close()

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