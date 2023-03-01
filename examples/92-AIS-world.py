import os

import geopandas

dat_dir = '/Users/Alten005/surfdrive/Kelvin/AIS-spire'
f_start = 'Global_24hours_AIS_24Oct2020_00000000000'

# galapagos, polygon...

for id in range(2):
    f_name = f_start + str(id+1) + '.csv'
    f_full = os.path.join(dat_dir, f_name)
    ais = geopandas.read_file(f_full)
    print(f_name)
print('done')