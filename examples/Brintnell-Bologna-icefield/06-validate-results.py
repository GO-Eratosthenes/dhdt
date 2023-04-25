"""
Validate results via indepdent elevation model, that stems from ultra-high
resolution photogrammetric elevation model
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from dhdt.generic.handler_www import get_file_from_www
from dhdt.generic.mapping_io import read_geo_image
from dhdt.postprocessing.glacier_tools import \
    volume2icemass, mass_changes2specific_glacier_hypsometries

# Brintnell-Bologna icefield (Northwest Territories)
MGRS_TILE = "09VWJ"

DATA_DIR = os.path.join(os.getcwd(), 'data') #"/project/eratosthenes/Data/"
DATA_URL = 'https://surfdrive.surf.nl/files/index.php/s/EX3UZ0o0Z6qW0NA'

REF_PATH = os.path.join(DATA_DIR, "PLEI", MGRS_TILE+'.tif')
RGI_PATH = os.path.join(DATA_DIR, "RGI", MGRS_TILE+'.tif')
DEM_PATH = os.path.join(DATA_DIR, "DEM", MGRS_TILE+'.tif')

def main():
    # do downloading if files are not present
    if not os.path.exists(REF_PATH):
        fname = get_file_from_www(DATA_URL,
                                  dump_dir=os.path.join(DATA_DIR, "PLEI"))
        os.rename(os.path.join(DATA_DIR, "PLEI", fname),
                  os.path.join(DATA_DIR, "PLEI", MGRS_TILE+'.tif'))

    ref_dat = read_geo_image(REF_PATH)[0]
    rgi_dat = read_geo_image(RGI_PATH)[0]
    dem_dat = read_geo_image(DEM_PATH)[0]

    # the Pleiades data is only available for a selection of the tile,
    # hence the masks need to align
    if type(ref_dat) in (np.ma.core.MaskedArray,):
        ref_dat.mask = np.logical_or(ref_dat.mask, np.isnan(ref_dat),
                                     rgi_dat.data==0)
    if type(dem_dat) in (np.ma.core.MaskedArray,):
        dem_dat.mask = np.logical_or(dem_dat.mask, ref_dat.mask)
    if type(rgi_dat) in (np.ma.core.MaskedArray,):
        rgi_dat.mask = np.logical_or(rgi_dat.mask, ref_dat.mask)


    mass_dat = volume2icemass(ref_dat, dem_dat, rgi_dat)
    Mb, rgi, z_bin = mass_changes2specific_glacier_hypsometries(
        mass_dat, dem_dat, rgi_dat, interval=100.)

#    Mb /= 1E3 # mwe
#    Mb /= 5 # yr
    print('.')
    plt.plot(np.tile(z_bin, (Mb.shape[0], 1)).T, Mb.T);

if __name__ == "__main__":
    main()
