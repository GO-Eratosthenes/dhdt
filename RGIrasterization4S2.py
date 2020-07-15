#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 09:12:39 2020

@author: Alten005
"""
import os
import osr
import ogr
import gdal

datPath = '/Users/Alten005/surfdrive/Eratosthenes/Denali/' 
rgiPath = datPath+'GIS/'
rgiName = '01_rgi60_alaska.shp'
imPath = datPath+'Data/'

# get tiles
s2List = [x for x in os.listdir(imPath) 
          if (os.path.isdir(os.path.join(imPath,x))) & (x[0:2]=='S2')]
tileList = [x.split('_')[5] for x in s2List]
tileList = list(set(tileList))

# find out how many utmzones are included
utmList = [int(x[1:3]) for x in tileList]
utmList = list(set(utmList))

for i in range(len(utmList)):
    # open the shapefile of RGI
    shp = ogr.Open(rgiPath+rgiName)
    lyr = shp.GetLayer()
    
    # transform to lat-lon to UTM
    spatRef = lyr.GetSpatialRef()
    driver = ogr.GetDriverByName('ESRI Shapefile')

    proj = osr.SpatialReference()
    proj.SetWellKnownGeogCS('WGS84')
    
    #####
    # only for Northern hemisphere, else: False
    # hence, look at tile Letters
    ####
    proj.SetUTM(utmList[i], True)
    
    coordTrans = osr.CoordinateTransformation(spatRef, proj)
    
    # create the output layer
    fnameShpUtm = rgiName[0:-4] + '_utm' + str(utmList[i]).zfill(2) + '.shp'
    if os.path.exists(rgiPath+fnameShpUtm):
        driver.DeleteDataSource(rgiPath+fnameShpUtm)
    outDataSet = driver.CreateDataSource(rgiPath+fnameShpUtm)
    outLyr = outDataSet.CreateLayer("RGIid", geom_type=ogr.wkbMultiPolygon)
    
    # add fields
    #lyrDefn = lyr.GetLayerDefn() # 4 str
    #lyrDefn.GetFieldDefn(0).SetType(2)
    lyrDefn = ogr.FieldDefn('RGIId', ogr.OFTInteger)
    
    outLyr.CreateField(lyrDefn.GetFieldDefn())
    #outLyr.CreateField(lyrDefn.GetFieldDefn(0))
    #outLyr.SetType(2)
    #outLyr.GetFieldDefn(0).SetType(2)
    # lyrDefn.GetFieldDefn(0).GetName()
    # get the output layer's feature definition
    outLyrDefn = outLyr.GetLayerDefn()
    
    feat = lyr.GetNextFeature()
    while feat:     
        geom = feat.GetGeometryRef() # get the input geometry     
        geom.Transform(coordTrans) # reproject the geometry from ll to UTM
        outFeat = ogr.Feature(outLyrDefn) # create a new feature
        outFeat.SetGeometry(geom) # set the geometry and attribute
        outFeat.SetField(0, int(feat.GetField(0).split('.')[1])) # assign RGIid, but removing leadin ID
        outLyr.CreateFeature(outFeat) # add the feature to the shapefile
        # dereference the features and get the next input feature
        outFeat = None
        feat = lyr.GetNextFeature()
    
    # Save and close the shapefiles
#    shp = None
#    outDataSet = None    
    proj.MorphToESRI() # provide correct prj-file for GIS
    file = open(rgiPath+fnameShpUtm[0:-4]+'.prj', 'w')
    file.write(proj.ExportToWkt())
    file.close()
    del spatRef, proj, file
    
    print('written ' + fnameShpUtm)
    del fnameShpUtm, outDataSet, outLyr, outLyrDefn, feat, geom, outFeat, lyr, lyrDefn, shp, driver

# loop through the tiles
for i in range(len(tileList)):
#    tileName = 'T05VNK';
    tileName = tileList[i]
    if not os.path.exists(rgiPath + tileName + '.tif'):
        # load RGI in correct UTM zone
        rgiNameUtm = rgiName[0:-4] + '_utm' + tileName[1:3] + '.shp'
        shp = ogr.Open(rgiPath+rgiNameUtm)
        lyr = shp.GetLayer()
        
        # find S2 data in the given tile
        s2Name = [x for x in s2List 
                  if x.split('_')[5]==tileName]
        s2Name = s2Name[0]
        s2Full = imPath+s2Name+'/'+tileName+'_'+s2Name.split('_')[2]+'_B04.jp2'
        
        
        # create raster environment
        S2src = gdal.Open(s2Full)
        driver = gdal.GetDriverByName('GTiff')
        target = driver.Create(rgiPath + tileName + '.tif', 
                               S2src.RasterXSize, S2src.RasterYSize,
                               1, gdal.GDT_UInt16)
        target.SetGeoTransform(S2src.GetGeoTransform())
        target.SetProjection(S2src.GetProjection())
        # convert polygon to raster
        gdal.RasterizeLayer(target, [1], lyr, None, options = ["ATTRIBUTE=RGIId"])
        gdal.RasterizeLayer(target, [1], lyr, None, burn_values=[1])
        
        print('written ' + tileName)
