# generic libraries
import pandas

# import geospatial libraries
import geopandas
from shapely.geometry import Polygon

dem_path = '/Users/Alten005/surfdrive/Eratosthenes/RedGlacier/Cop-DEM-GLO-30/'
ftp_file = 'mapping.csv'
file_path = dem_path+ftp_file

out_file = 'DGED-30.geojson'

# load file with ftp-addresses
df = pandas.read_csv(file_path, delimiter=";", header=0)
df_org = df.copy()

# creating dictionary for trans table
trans_dict ={"N": "+", "S": "-", "E": "+", "W": "-"}
# creating translate table from dictionary
trans_table ="NSEW".maketrans(trans_dict)
# translating through passed transtable
df["Lat"] = df["Lat"].str.translate(trans_table)
df["Long"] = df["Long"].str.translate(trans_table) 

# naming is the lower left corner
ll = df[["Long","Lat"]].astype(float)
def f(x):    
    return Polygon.from_bounds(x[0], x[1], x[0]+1, x[1]+1)  
geom = ll.apply(f, axis=1)

# include geometry and make a geopandas dataframe
gdf = geopandas.GeoDataFrame(df_org, geometry=geom)
gdf.set_crs(epsg=4326)

file_path = dem_path+out_file
gdf.to_file(file_path, driver='GeoJSON')
#gdf.to_file(dem_path+'DGED-30.shp')
