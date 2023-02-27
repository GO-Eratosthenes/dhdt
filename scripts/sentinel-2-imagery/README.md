# Download Sentinel-2 Imagery

## Introduction

In [this notebook](../../docs/tutorials/sentinel-2-imagery.ipynb), we have searched for the Sentinel-2 images that 
include a predefined area-of-interest (the Brintnell-Bologna icefield) via the [Copernicus Data Access Hub API](https://scihub.copernicus.eu/)
and the [Sentinelsat tool](https://github.com/sentinelsat/sentinelsat). Metadata from the scenes that match the search 
have been saved in GeoJSON format. 

We provide here scripts and instructions to:

* Convert the scenes' metadata from GeoJSON to the SpatioTemporal Asset Catalog (STAC) format, linking the most relevant 
  Sentinel-2 data products to the corresponding files on [Google Cloud Storage (GCS)](https://cloud.google.com/storage/docs/public-datasets/sentinel-2).

* Download the data files from GCS and store these alongside the corresponding scene's metadata. 

## Requirements

The following required libraries can be installed with `pip`:

```shell
pip install geopandas pystac stactools stactools-sentinel2
```

## Convert GeoJSON search results to STAC

Run the Python script [sentinel-2-STAC.py](./sentinel-2-STAC.py) with the `create` positional argument to create a 
catalog with links to the GCS assets, e.g.:

```shell
# Create Sentinel-2 L1C data catalog
python sentinel-2-STAC.py --catalog-path ./data/sentinel-2/sentinel2-l1c create --from-geojson ./data/sentinel-2/sentinel2-l1c.json --description 'Sentinel-2 L1C scenes of the Brintnell-Bologna icefield' --template '${year}/${month}/${day}'
# Create Sentinel-2 L2A data catalog
python sentinel-2-STAC.py --catalog-path ./data/sentinel-2/sentinel2-l2a create --from-geojson ./data/sentinel-2/sentinel2-l2a.json --description 'Sentinel-2 L2A scenes of the Brintnell-Bologna icefield' --template '${year}/${month}/${day}'
```

## Download imagery

Run the same script with the `download` positional argument to retrieve (a subset of) the catalogs' assets:

```shell
# Download Sentinel-2 L1C data (few bands and metadata files)
python sentinel-2-STAC.py --catalog-path ./data/sentinel-2/sentinel2-l1c/catalog.json  download --assets blue green red nir product_metadata granule_metadata inspire_metadata datastrip_metadata sensor_metadata_B02 sensor_metadata_B03 sensor_metadata_B04 sensor_metadata_B08
# Download Sentinel-2 L2A data (only scene classification layer)
python sentinel-2-STAC.py --catalog-path ./data/sentinel-2/sentinel2-l2a/catalog.json  download --assets scl
```
