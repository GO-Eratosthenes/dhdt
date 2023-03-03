# functions of use for the CopernicusDEM data

# generic libraries
import os
import pathlib

# import geospatial libraries
import rioxarray

from rioxarray.merge import merge_arrays # had some troubles since I got:
    # AttributeError: module 'rioxarray' has no attribute 'merge'
from rasterio.enums import Resampling
from affine import Affine
import pyproj

# import local functions
from dhdt.generic.handler_www import get_file_from_www
from dhdt.generic.handler_sentinel2 import get_crs_from_mgrs_tile
from dhdt.auxilary.handler_mgrs import normalize_mgrs_code, get_geom_for_tile_code, get_bbox_from_tile_code

def make_copDEM_mgrs_tile(mgrs_tile, out_path=None, geom_path=None, resolution='30'):
    """

    Parameters
    ----------
    mgrs_tile : string
        MGRS code of the tile of interest
    out_path : string
        location where the DEM file can be written
    geom_path : string
        Path to the geometric metadata
    resolution : string
        Resolution of the DEM ('30' or '90')

    Returns
    -------
    cop_dem_out : string
        location and filename of the raster

    Notes
    -----
    The following acronyms are used:

    - CDS : Copernicus data service
    - CopDEM : Copernicus DEM
    - CRS : coordinate reference system
    - DEM : digital elevation model
    - DGED : defence gridded elevation data
    - DTED : digital terrain elevation data
    - GLO : global
    - MGRS : US Military Grid Reference System
    - SSO : single sign on
    """
    mgrs_tile = normalize_mgrs_code(mgrs_tile)
    bbox = get_bbox_from_tile_code(mgrs_tile, geom_path)
    crs = pyproj.CRS.from_epsg(32605)
    shape = (10980, 10980)
    transform = Affine.from_gdal(
        399960.0, 10.0, 0.0, 6700020.0, 0.0, -10.0
    )
    
    dem_tiles = [round(i) for i in bbox]
    url_list = get_cds_urls(dem_tiles, resolution)

    cop_dem = download_and_mosaic(url_list, out_path, shape, crs, transform)
    cop_dem_out = os.path.join(out_path, 'COP-DEM-' + mgrs_tile + '.tif')
    cop_dem.rio.to_raster(cop_dem_out, dtype=cop_dem.dtype)
    return cop_dem_out

def get_cds_urls(dem_tiles, resolution):

    NS_trans_dict = {"+": "N", "-": "S"}
    EW_trans_dict = {"+": "E", "-": "W"}
    NS_trans_table = "NS".maketrans(NS_trans_dict)
    EW_trans_table = "EW".maketrans(EW_trans_dict)

    url_list = []
    for i in range(dem_tiles[0], dem_tiles[1]+1):
        ew = str(i)
        if i>0: ew = '+'+str(i)
        for j in range(dem_tiles[2], dem_tiles[3]+1):
            ns = str(j)
            if j>0: ns = '+'+str(j)
            name = 'Copernicus_DSM_COG_10_'+ ns.translate(NS_trans_table) +'_00_' + ew.translate(EW_trans_table) +'_00_DEM'
            url = 'https://s3.eu-central-1.amazonaws.com/copernicus-dem-'+ resolution +'m/'+ name + '/' + name + '.tif'
            url_list.append(url)

    return url_list


def download_and_mosaic(url_list, tmp_path, shape, crs, transform):
    """ Download Copernicus DEM tiles and create mosaic according to satellite
    imagery tiling scheme
    
    file_list : list with strings
        list of DEM tile filenames
    tmp_path : string
        work path where to download and untar DEM tiles
    shape : list
        shape of raster
    crs : string
        coordinate reference string
    transform :

    Returns
    -------
    dem_clip : DataArray object
        retiled DEM
    """
    print('Great you use CopernicusDEM, if you do use this data for ' +
          'products, publications, repositories, etcetera. \n' +
          'Please make sure you cite via the following doi: ' +
          'https://doi.org/10.5270/ESA-c5d3d65')  
    for url in url_list:
        get_file_from_www(url, tmp_path)

        
    dem_tiles_filename = pathlib.Path(tmp_path).glob("*_DEM.tif")
    dem_clip = mosaic_tiles(dem_tiles_filename, shape, crs, transform)
        
    # sometimes out of bound tiles are still present,
    # hence rerun a clip to be sure
    # dem_clip = dem_clip.rio.clip_box(*bbox)
    return dem_clip

def mosaic_tiles(dem_tiles_filenames, shape, crs, transform):
    """ given a selection of elevation models, mosaic them together

    Parameters
    ----------
    dem_tiles_filenames : list with strings
        list of DEM tile urls
    shape : list
        
    crs : string
        coordinate reference string
    transform : 

    Returns
    -------
    merged : DataArray object
        mosaiced DEM
    """
    merged = None
    for f in dem_tiles_filenames:
        dem_tile = rioxarray.open_rasterio(f)
        # reproject to image CRS
        dem_tile_xy = dem_tile.rio.reproject(crs,
                                             transform=transform,
                                             resampling=Resampling.lanczos,
                                             shape=shape)
        if merged is None:
            merged = dem_tile_xy
        else:
            merged = merge_arrays([merged, dem_tile_xy])
    return merged