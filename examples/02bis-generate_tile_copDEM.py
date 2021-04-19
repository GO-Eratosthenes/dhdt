import os
import getpass

import pathlib
import tempfile

import tarfile

import geopandas
import pystac
import rioxarray
import rioxarray.merge
import rasterio.transform
import stac2dcache

from shapely.geometry import Polygon
from stac2dcache.utils import get_asset
from eratosthenes.generic.handler_www import get_file_from_ftps

# start functions
def read_catalog(url):
    """
    Read STAC catalog from URL
    
    :param url: urlpath to the catalog root
    :return: PySTAC Catalog object
    """
    url = url if url.endswith("catalog.json") else f"{url}/catalog.json"
    catalog = pystac.Catalog.from_file(url)
    return catalog

def get_sentinel2_tile_id(item):
    """
    Construct the tile ID for a Sentinel-2 STAC item
    
    :param item: PySTAC Item object
    :return: tile ID
    """
    return "".join([
        str(item.properties[k]) for k in TILE_ID_KEYS
    ])

def get_itersecting_DEM_tile_names(index, geometry):
    """ 
    Find the DEM tiles that intersect the geometry and extract the
    filenames from the index. NOTE: the geometries need to be in the 
    same CRS!
    
    :param index: DEM tile index (GeoDataFrame)
    :param geometry: shapely geometry object
    :returl: filename array
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
        
        # first transform bbox from intended to source projection,
        # in this way, clipping is possible and less transformation is needed        
        for f in dem_tiles_filename:
            dem_tile = rioxarray.open_rasterio(f)
            # reproject to image CRS
            dem_tile_xy = dem_tile.rio.reproject(crs, transform=transform)
            
            bbox_xy = dem_tile_xy.rio.bounds()
            bbox_xy = (min(bbox[0],bbox_xy[0]), min(bbox[1],bbox_xy[1]),
                       max(bbox[0],bbox_xy[0]), max(bbox[1],bbox_xy[1]))
            # extend area
            dem_tile_xy = dem_tile_xy.rio.pad_box(*bbox_xy)
            # crop area within tile
            dem_tile_xy = dem_tile_xy.rio.clip_box(*bbox)
            if 'dem_clip' not in locals():
                dem_clip = dem_tile    
            else:
                dem_clip = rioxarray.merge.merge_arrays([dem_clip, 
                                                        dem_tile_xy])                
    return dem_clip

def get_SSO_id(fpath):
    with open(fpath, 'r') as file:
        lines = file.read().splitlines()
    ssoid, pw = lines[0], lines[1]    
    return ssoid, pw

# start script

catalog_url = ("https://webdav.grid.surfsara.nl:2880"
               "/pnfs/grid.sara.nl/data/eratosthenes/"
               "disk/red-glacier_sentinel-2")
collection_id = "sentinel-s2-l1c"
dem_index_url = (
    "https://webdav.grid.surfsara.nl:2880"
    "/pnfs/grid.sara.nl/data/eratosthenes/"
    "disk/GIS/Elevation/DGED-30.geojson"
    )
tmp_path = "./"
dem_tiles_url = (
    "https://webdav.grid.surfsara.nl:2880"
    "/pnfs/grid.sara.nl/data/eratosthenes/"
    "disk/CopernicusDEM_tiles_sentinel-2"
    )

# configure connection to dCache
username = getpass.getuser()
if username=='Alten005':
    dcache = stac2dcache.configure(
        filesystem="dcache", 
        token_filename=os.path.expanduser("~/Eratosthenes/HPC/macaroon.dat")
        )
else:
    dcache = stac2dcache.configure(
        filesystem="dcache", 
        token_filename="macaroon.dat"
        )
    
# read DEM index
with dcache.open(dem_index_url) as f:
    dem_index = geopandas.read_file(f)
    
# read image catalog
catalog = read_catalog(catalog_url)
subcatalog = catalog.get_child(collection_id)    
    
TILE_ID_KEYS = [
    "sentinel:utm_zone", 
    "sentinel:latitude_band", 
    "sentinel:grid_square"
    ]     

# loop over catalog, look for all the tiles presents
tiles = {}
for item in subcatalog.get_all_items():
    tile_id = get_sentinel2_tile_id(item)
    if tile_id not in tiles:
        tiles[tile_id] = item
   
# loop over identified tiles
if username=='Alten005':
    sso, pw = get_SSO_id(os.path.expanduser("~/Eratosthenes/HPC/ESA_SSO-ID"))
else:
    sso = 'please_fill_in-your_ESA_sso-id'
    pw = 'test'
    
cds_url = 'cdsdata.copernicus.eu:990'
cds_path = '/datasets/COP-DEM_GLO-30-DGED_PUBLIC/2019_1/'

for tile_id, item in tiles.items():

    da = get_asset(
        catalog,
        asset_key="B02",
        item_id=item.id,
        filesystem=dcache,
        load=False
    )
    bbox = da.rio.bounds()
    crs = da.spatial_ref.crs_wkt
    transform = da.rio.transform()
    
    tile_geometry = Polygon.from_bounds(*bbox)
    tars = get_itersecting_DEM_tile_names(
        dem_index.to_crs(crs),
        tile_geometry
    )
    
    dem = download_and_mosaic_through_ftps(
        tars, tmp_path, cds_url, cds_path,
        sso, pw, bbox, crs, transform
    )
    
    # save raster file and upload it
    output_file = f"{tile_id}.tif"
    dem.rio.to_raster(output_file)
    dcache.upload(output_file, f"{dem_tiles_url}/{output_file}")




