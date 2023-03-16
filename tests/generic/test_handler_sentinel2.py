from dhdt.generic.handler_sentinel2 import get_generic_s2_raster

TESTDATA_DIR = 'testdata/'

# The MGRS file
MGRS_INDEX_FILE = f'{TESTDATA_DIR}/MGRS/mgrs_tiles_tiny.geojson'


def test_get_generic_s2_raster_returns_correct_crs():
    tile_codes = ['07MHV', '01CCV', '41UQV', '60MXA']
    utm_zones = ['7S', '1S', '41N', '60S']
    for tile_code, utm_zone in zip(tile_codes, utm_zones):
        _, crs = get_generic_s2_raster(
            tile_code=tile_code,
            tile_path=MGRS_INDEX_FILE
        )
        assert crs.utm_zone == utm_zone
