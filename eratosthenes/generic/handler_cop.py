# functions of use for the CopernicusDEM data

import os
import tarfile
import pathlib
import tempfile

import rioxarray
from rioxarray.merge import merge_arrays # had some troubles since I got:
    # AttributeError: module 'rioxarray' has no attribute 'merge'
from rasterio.enums import Resampling

from .handler_www import get_file_from_ftps

def get_itersecting_DEM_tile_names(index, geometry):
    """
    Find the DEM tiles that intersect the geometry and extract the
    filenames from the index. NOTE: the geometries need to be in the 
    same CRS!
    
    Parameters
    ----------
    index : dtype=GeoDataFrame
        DEM tile index.
    geometry : shapely geometry object
        polygon of interest.

    Returns
    -------
    fname
        file name of tiles of interest.
    """    
    # sometimes points are projected to infinity
    # this is troublesome for intestections
    # hence remove these instances
    out_of_bounds = index['geometry'].area.isna()
    index = index[~out_of_bounds]
    
    mask = index.intersects(geometry)
    index = index[mask]
    return index['CPP filename']

def copDEM_files(members):
    for tarinfo in members:
        if tarinfo.name.endswith('DEM.tif'):
            yield tarinfo

def download_and_mosaic_through_ftps(file_list, tmp_path, cds_url, cds_path,
                                     sso, pw, bbox, crs, transform):
    """
    Download DEM tiles and create mosaic according to satellite imagery tiling
    scheme
    
    :param url_list: list of DEM tile filenames
    :param tmp_path: work path where to download and untar DEM tiles
    :param bbox: bound box (xmin, ymin, xmax, ymax)
    :param crs: coordinate reference system of the DEM tile
    :param transform: Affine transform of the DEM tile
    :return: retiled DEM (DataArray object) 
    """
    with tempfile.TemporaryDirectory(dir=tmp_path) as tmpdir:
    
        for file_name in file_list:
            get_file_from_ftps(cds_url, sso, pw, cds_path, file_name, tmpdir)

        tar_tiles_filenames = [f for f in os.listdir(tmpdir) if f.endswith('.tar')]
        for tar_fname in tar_tiles_filenames:
            tar_file = tarfile.open(os.path.join(tmpdir, tar_fname), mode="r|")
            tar_file.extractall(members=copDEM_files(tar_file), 
                                path=tmpdir)
            tar_file.close()
            
            
        dem_tiles_filename = pathlib.Path(tmpdir).glob("**/*_DEM.tif")
        dem_clip = mosaic_tiles(dem_tiles_filename, bbox, crs, transform)
          
        # sometimes out of bound tiles are still present,
        # hence rerun a clip to be sure
        # dem_clip = dem_clip.rio.clip_box(*bbox)
    return dem_clip

def mosaic_tiles(dem_tiles_filenames, bbox, crs, transform):
    # first transform bbox from intended to source projection,
    # in this way, clipping is possible and less transformation is needed        
    for f in dem_tiles_filenames:
        dem_tile = rioxarray.open_rasterio(f)
        dem_tile = dem_tile.rio.write_nodata(-9999)
        # reproject to image CRS
        dem_tile_xy = dem_tile.rio.reproject(crs, transform=transform, 
                                             resampling=Resampling.lanczos)
        
        bbox_xy = dem_tile_xy.rio.bounds()
        bbox_xy = (min(bbox[0],bbox_xy[0]), min(bbox[1],bbox_xy[1]),
                    max(bbox[2],bbox_xy[2]), max(bbox[3],bbox_xy[3])) # spatial OR

        # extend area
        dem_tile_xy = dem_tile_xy.rio.pad_box(minx=bbox_xy[0], \
                                              miny=bbox_xy[2], \
                                              maxx=bbox_xy[1], \
                                              maxy=bbox_xy[3], \
                                              constant_values=-9999) 
        
        # crop area within tile
        dem_tile_xy = dem_tile_xy.rio.clip_box(minx=bbox[0], \
                                               miny=bbox[2], \
                                               maxx=bbox[1], \
                                               maxy=bbox[3])
                                               
        if 'dem_clip' not in locals():
            dem_clip = dem_tile_xy    
        else:
            dem_clip = merge_arrays([dem_clip, 
                                     dem_tile_xy], 
                                    nodata=-9999)   
    return dem_clip