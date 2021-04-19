import geopandas
import ftps

from eratosthenes.generic.handler_www import get_file_from_ftps, unpack_tar_file

dem_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Cop-DEM-GLO-30/'
cop_file = 'DGED-30.geojson'
file_path = dem_path+cop_file

gdf = geopandas.read_file(file_path)

urls = gdf['downloadURL']
tars = gdf['CPP filename']

sso = 'test'
pw = 'test'

cds_url = 'cdsdata.copernicus.eu:990'
dem_url =urls[0]
tar_name = tars[0]
cds_path = '/datasets/COP-DEM_GLO-30-DGED_PUBLIC/2019_1/'

get_file_from_ftps(cds_url, sso, pw, cds_path, tar_name, dem_path)


client = ftps.FTPS('ftps://' + sso+':' +pw+ '@cdsdata.copernicus.eu:990')
client.list()
client.download(cds_path + tar_name, "tmp.tar")

