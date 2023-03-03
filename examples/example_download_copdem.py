from dhdt.auxilary.handler_copernicusdem import \
    make_copDEM_mgrs_tile

mgrs_tile = '05VMG'

make_copDEM_mgrs_tile(mgrs_tile, out_path='../data/copDEM', geom_path='../data/MGRS/sentinel2_tiles_world.geojson')