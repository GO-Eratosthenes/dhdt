import urllib.request
import geopandas as gpd
from shapely.geometry import Polygon
from dhdt.auxilary import handler_mgrs

# A subset of original MGRS, sampling every 1500 entries
TESTDATA_KML = 'testdata/MGRS/mgrs_tiles_tiny.kml'


def test_mgrs_kml_url_valid():
    assert urllib.request.urlopen(handler_mgrs.MGRS_TILING_URL).code == 200


def test_kml_to_gdf_output_all_polygon():
    gdf = handler_mgrs._kml_to_gdf(TESTDATA_KML)
    assert all([isinstance(geom, Polygon) for geom in gdf["geometry"]])


def test_kml_to_gdf_output_same_crs():
    gdf_kml = gpd.read_file(TESTDATA_KML)
    gdf = handler_mgrs._kml_to_gdf(TESTDATA_KML)
    assert gdf.crs.equals(gdf_kml.crs)
