import os

from eratosthenes.processing.coupling_tools import couple_pair
from eratosthenes.processing.gis_tools import make_casting_couple_shapefile

fpath1 = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-15-10-2019/', \
                      'shadow.tif')
fpath2 = os.path.join('/Users/Alten005/GO-eratosthenes/start-code/examples/S2-25-10-2019/', \
                      'shadow.tif')
#dat_path = '/Users/Alten005/GO-eratosthenes/start-code/examples/'
dir_path = os.path.dirname(os.path.realpath(__file__)) 
if os.getcwd()!=dir_path: 
    os.chdir(dir_path) # change to directory where script is situated


match_method='norm_corr'#'aff_of'
subpix_method='weighted'
xy_1, xy_2, casters, dh, crs = couple_pair(fpath1, fpath2, 
                                      None, None, None,
                                      rgi_id=None, reg='none', prepro=None,
                                      match=match_method, boi=1,
                                      subpix=subpix_method,
                                      processing='shadow')

time_1, time_2 = 'S2-15-10-2019', 'S2-25-10-2019'

dh_fname = ('dh-' + time_1 +'-'+ time_2 + '-' + match_method +'.txt')
    
f = open(dh_fname, 'w')
# write projection data, e.g.: EPSG-32605
f.write(crs + '\n')
for k in range(dh.shape[0]):
    line = ('{:+8.2f}'.format(   xy_1[k, 0]) + ' ' + 
            '{:+8.2f}'.format(   xy_1[k, 1]) + ' ' +
            '{:+8.2f}'.format(   xy_2[k, 0]) + ' ' +
            '{:+8.2f}'.format(   xy_2[k, 1]) + ' ' +
            '{:+8.2f}'.format(casters[k, 0]) + ' ' +
            '{:+8.2f}'.format(casters[k, 1]) + ' ' +
            '{:+4.3f}'.format(     dh[k]))
    f.write(line + '\n')
f.close()

make_casting_couple_shapefile(os.path.join(dir_path, dh_fname))
print('done')