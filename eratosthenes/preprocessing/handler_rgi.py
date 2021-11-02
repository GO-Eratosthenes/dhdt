# functions to work with the Randolph Glacier Inventory
import os
import numpy as np

from osgeo import ogr

from ..generic.handler_www import url_exist, get_zip_file

def rgi_code2url(rgi_num):
    """
    Given an RGI-number, give the location of its shapefiles on GLIMS-website
    """    
    urls = (
        'rgi60_files/01_rgi60_Alaska.zip',
        'rgi60_files/02_rgi60_WesternCanadaUS.zip',
        'rgi60_files/03_rgi60_ArcticCanadaNorth.zip',
        'rgi60_files/04_rgi60_ArcticCanadaSouth.zip',
        'rgi60_files/05_rgi60_GreenlandPeriphery.zip',
        'rgi60_files/06_rgi60_Iceland.zip',
        'rgi60_files/07_rgi60_Svalbard.zip',
        'rgi60_files/08_rgi60_Scandinavia.zip',
        'rgi60_files/09_rgi60_RussianArctic.zip',
        'rgi60_files/10_rgi60_NorthAsia.zip',
        'rgi60_files/11_rgi60_CentralEurope.zip',
        'rgi60_files/12_rgi60_CaucasusMiddleEast.zip',
        'rgi60_files/13_rgi60_CentralAsia.zip',
        'rgi60_files/14_rgi60_SouthAsiaWest.zip',
        'rgi60_files/15_rgi60_SouthAsiaEast.zip',
        'rgi60_files/16_rgi60_LowLatitudes.zip',
        'rgi60_files/17_rgi60_SouthernAndes.zip',
        'rgi60_files/18_rgi60_NewZealand.zip',
        'rgi60_files/19_rgi60_AntarcticSubantarctic.zip'
        )
    rgi_url = ()
    for i in range(0,len(rgi_num)):
        rgi_url += ('https://www.glims.org/RGI/' + urls[rgi_num[i]-1],)
    return rgi_url

def which_rgi_region(rgi_path,poi):
    '''
    Discover which Randolph Glacier Inventory (RGI) region is used 
    for a given coordinate
    '''

    rgi_file = '00_rgi60_O1Regions.shp'
    
    if np.ndim(poi)==1:
        # lon in range -180:180
        poi[1] = (poi[1] % 360)-360
        poi = np.flip(poi) # given in lon lat....
        
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(poi[0], poi[1])
        

        # Get the RGI gegions
        rgiShp = ogr.Open(os.path.join(rgi_path, rgi_file))
        rgiLayer = rgiShp.GetLayer()

        rgi_list = ()
        # loop through the regions and see if there is an intersection
        for i in range(0, rgiLayer.GetFeatureCount()):
            # Get the input Feature
            rgiFeature = rgiLayer.GetFeature(i)
            geom = rgiFeature.GetGeometryRef()
            
            if point.Within(geom):
                rgi_list += (rgiFeature.GetField('RGI_CODE'),)
                print('it seems the region you are interest in falls within '+
                      rgiFeature.GetField('FULL_NAME'))
                
    else: # polygon based
        print('not yet implemented!')

    
    if len(rgi_list)==0:
        print('selection not in Randolph glacier region')
    else:
        rgi_url = rgi_code2url(rgi_list)
        # does RGI region exist?
        
        rgi_file = ()
        for i in range(0, len(rgi_url)):
            rgi_reg = rgi_url[i]
            rgi_zip = rgi_reg.split('/')[-1]
            rgi_shp = rgi_zip[:-4] + '.shp'
            
            dir_entries = os.scandir(rgi_path)
            found_rgi = False
            for entry in dir_entries:
                if entry.name == rgi_shp:
                    print(entry.name + ' is found in the folder')
                    rgi_file += (rgi_shp, )
                    found_rgi = True
                
            # appearantly the file is not present
            if (url_exist(rgi_reg) and not(found_rgi)):
                print('downloading RGI region')
                get_zip_file(rgi_reg, rgi_path)
                rgi_file += (rgi_shp, )
            
    return rgi_file