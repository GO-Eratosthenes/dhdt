# generic libraries
import os

import tarfile
import zipfile
import bz2
import urllib.request

import ftps # for Copernicus FTP-download
import requests # for NASA Earthdata Login
import warnings

# geospatial libaries
from osgeo import gdal


def get_file_from_ftps(url, user, password,
                       file_path, file_name, dump_dir=os.getcwd()):
    """ Downloads a file from a ftps-server

    Paramters
    ---------
    url : string
        server address
    user : string
        username
    password : string
        password for access
    file_path : string
        location on the server
    file_name : string
        name of the file
    dump_dir : string
        path to place the content
    """
    if dump_dir[-1]!=os.sep:
        dump_dir += os.sep
    if not os.path.isdir(dump_dir): os.makedirs(dump_dir)

    client = ftps.FTPS('ftps://' +user+ ':' +password+ '@' +url)
    client.list()
    client.download( os.path.join(file_path, file_name),
                     os.path.join(dump_dir, file_name))
    return


def get_file_from_www(full_url, dump_dir=os.getcwd(), overwrite=False):
    """ Downloads a file from a location given by the url

    Parameters
    ----------
    full_url : string
        depository where the file is situated
    dump_dir : string
        folder location where the data needs to be unpacked
    overwrite : bool
        if True and file exists, download it again, otherwise do nothing

    Returns
    -------
    file_name : string
        name of the downloaded file

    See Also
    --------
    get_file_from_protected_www
    """
    assert isinstance(full_url, str), 'please provide a string'
    assert isinstance(dump_dir, str), 'please provide a string'

    file_name = full_url.split('/')[-1]
    file_path = os.path.join(dump_dir, file_name)

    if not os.path.isfile(file_path) or overwrite:
        assert url_exist(full_url), f"Non-existing URL: {full_url}"
        os.makedirs(dump_dir, exist_ok=True)
        # download data
        urllib.request.urlretrieve(full_url, file_path)
    return file_name


def get_file_from_protected_www(full_url, dump_dir=os.getcwd(),
                                user=None, password=None):
    """ Some data is situated in password protected repositories, this function

    Parameters
    ----------
    full_url : string
        server address
    dump_dir : string
        path to place the content
    user : string
        username
    password : string
        password for access

    See Also
    --------
    get_file_from_www
    """
    assert user is not None, 'please provide username'
    assert password is not None, 'please provide a password'
    if not os.path.isdir(dump_dir): os.makedirs(dump_dir)

    file_name = full_url.split('/')[-1]
    with requests.Session() as session:
        r1 = session.request('get', full_url)
        r = session.get(r1.url, auth=(user, password))
        if r.ok:
            with open(os.path.join(dump_dir, file_name), 'wb') as f:
                f.write(r.content)
        else:
            warnings.warn("Something might have gone wrong while downloading")
            return
    return

def url_exist(file_url):
    """ Check if an url exist

    Parameters
    ----------
    file_url : string
        url of www location

    Returns
    -------
    verdict : dtype=boolean
        verdict if present
    """
    return urllib.request.urlopen(file_url).code == 200


def get_bz2_file(bz2_url, dump_dir=os.getcwd(), overwrite=False):
    """ Level-3 data of Terra ASTER is packed into bz2-files. This function
    downloads and unpacks such data

    Parameters
    ----------
    bz2_url : string
        depository where the file is situated
    dump_dir : string
        folder location where the data needs to be unpacked
    overwrite : bool
        if True and file exists, download it again, otherwise do nothing

    Returns
    -------
    bz2_names : list
        list of strings of file names within the compressed folder
    """
    # download data
    zip_name = get_file_from_www(bz2_url, dump_dir, overwrite)

    # decompress
    zipfile = bz2.BZ2File(os.path.join(dump_dir, zip_name))
    data = zipfile.read()
    tar_name = zip_name[:-4]
    with open(os.path.join(dump_dir, tar_name), 'wb') as new_file:
        new_file.write(data)

    # extract out of tar
    tar_file = tarfile.open(os.path.join(dump_dir, tar_name), mode="r")
    tar_file.extractall(path=dump_dir)
    bz2_names = tar_file.getnames()
    return bz2_names


def get_tar_file(tar_url, dump_dir=os.getcwd(), overwrite=False):
    """ Downloads and unpacks compressed folder

    Parameters
    ----------
    tar_url : string
        url of world wide web location
    dump_dir : string
        path to place the content
    overwrite : bool
        if True and file exists, download it again, otherwise do nothing

    Returns
    -------
    tar_paths : list
        list of strings of file paths within the compressed folder
    """
    tar_file = get_file_from_www(tar_url, dump_dir, overwrite)
    tar_path = os.path.join(dump_dir, tar_file)
    with tarfile.open(name=tar_path, mode="r:gz") as tf:
        tar_names = tf.getnames()
        tf.extractall(path=dump_dir)
    return tar_names


def get_zip_file(zip_url, dump_dir=os.getcwd(), overwrite=False):
    """ Downloads and unpacks compressed folder

    Parameters
    ----------
    zip_url : string
        url of world wide web location
    dump_dir : string
        path to place the content
    overwrite : bool
        if True and file exists, download it again, otherwise do nothing

    Returns
    -------
    zip_names : list
        list of strings of file names within the compressed folder
    """
    zip_file = get_file_from_www(zip_url, dump_dir, overwrite)
    with zipfile.ZipFile(os.path.join(dump_dir, zip_file)) as zf:
        zip_names = zf.namelist()
        zf.extractall(path=dump_dir)
    return zip_names


def get_file_and_extract(full_url, dump_dir=os.getcwd(), overwrite=False):
    """ Downloads and unpacks compressed folder

    Parameters
    ----------
    full_url : string
        url of world wide web location
    dump_dir : string
        path to place the content
    overwrite : bool
        if True and file exists, download it again, otherwise do nothing

    Returns
    -------
    file_names : list
        list of strings of file names within the compressed folder
    """
    if any(full_url.endswith(ext) for ext in ('.tar.gz', '.tgz')):
        f = get_tar_file
    elif full_url.endswith('.zip'):
        f = get_zip_file
    elif full_url.endswith('.bz2'):
        f = get_bz2_file
    else:
        raise IndexError(f'Unknown extension: {full_url}')
    return f(full_url, dump_dir, overwrite)



def bulk_download_and_mosaic(url_list, dem_path, sat_tile, bbox, crs, new_res=10):

    for i in range(len(url_list)):
        gran_url = url_list[i]
        gran_url_new = change_url_resolution(gran_url,new_res)
        
        # download and integrate DEM data into tile
        print('starting download of DEM tile')
        if url_exist(gran_url_new):
            tar_names = get_tar_file(gran_url_new, dem_path)
        else:
            tar_names = get_tar_file(gran_url, dem_path)
        print('finished download of DEM tile')
            
        # load data, interpolate into grid
        dem_name = [s for s in tar_names if 'dem.tif' in s]
        if i ==0:
            dem_new_name = sat_tile + '_DEM.tif'
        else:
            dem_new_name = dem_name[0][:-4]+'_utm.tif'
        
        ds = gdal.Warp(os.path.join(dem_path, dem_new_name), 
                       os.path.join(dem_path, dem_name[0]), 
                       dstSRS=crs,
                       outputBounds=(bbox[0], bbox[2], bbox[1], bbox[3]),
                       xRes=new_res, yRes=new_res,
                       outputType=gdal.GDT_Float64)
        ds = None
        
        if i>0: # mosaic tiles togehter
            merge_command = ['python', 'gdal_merge.py', 
                             '-o', os.path.join(dem_path, sat_tile + '_DEM.tif'), 
                             os.path.join(dem_path, sat_tile + '_DEM.tif'), 
                             os.path.join(dem_path, dem_new_name)]
            my_env = os.environ['CONDA_DEFAULT_ENV']
            os.system('conda run -n ' + my_env + ' '+
                      ' '.join(merge_command[1:]))
            os.remove(os.path.join(dem_path,dem_new_name))
            
        for fn in tar_names:
            os.remove(os.path.join(dem_path,fn))


def change_url_resolution(url_string,new_res):
    """ the file name can have the spatail resolution within, this function
    replaces this string

    Paramters
    ---------
    url_string : string
        url of world wide web location
    new_res : integer
        new resolution (10, 32, ...)

    Returns
    -------
    url_string : string
        url of new world wide web location
    """
    
    # get resolution
    props = url_string.split('_')
    for i in range(1,len(props)):
        if props[i][-1] == 'm':
            old_res = props[i]
            props[i] = str(new_res)+'m'
    
    if (old_res=='2m') & (new_res==10):
        # the files are subdivided in quads
        props = props[:-4]+props[-2:]
    
    url_string_2 = '_'.join(props)
    
    folders = url_string_2.split('/')
    for i in range(len(folders)):
        print
        if folders[i] == old_res:
            folders[i] = str(new_res)+'m'
    
    gran_url_new = '/'.join(folders)
    return gran_url_new


def reduce_duplicate_urls(url_list):
    """ because the shapefiles are in 2 meter, the tiles are 4 fold, therfore
    make a selection, to bypass duplicates

    Parameters
    ----------
    url_list : list
          list of strings with url's of www locations

    Returns
    -------
    url_list : list
        reduced list of strings with url's of www location
    """
    tiles = ()
    for i in url_list: 
        tiles += (i.split('/')[-2],)
    uni_set = set(tiles)
    ids = []
    for i in range(len(uni_set)):
        idx = tiles.index(uni_set.pop())
        ids.append(idx)
    url_list = [url_list[i] for i in ids]    
#    print('reduced to '+str(len(url_list))+ ' elevation chips')
    return url_list
