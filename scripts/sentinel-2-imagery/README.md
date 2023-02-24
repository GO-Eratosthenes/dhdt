# Download Sentinel-2 Imagery

## Introduction

In [this notebook](../../docs/tutorials/sentinel-2-imagery.ipynb), we have searched for the Sentinel-2 images that 
includes a predefined area-of-interest (the Brintnell-Bologna icefield) via the [Copernicus Data Access Hub API]()
and the [Sentinelsat tool](). Metadata from the scenes that match the search have been saved in GeoJSON format. 

We provide here scripts and instructions to:

* Convert the scenes' metadata from GeoJSON to the SpatioTemporal Asset Catalog (STAC) format, linking the most relevant 
  Sentinel-2 data products to the corresponding files on [Google Cloud Storage (GCS)](https://cloud.google.com/storage/docs/public-datasets/sentinel-2).

* Download the files from GCS and store these alongside the corresponding scene's metadata. 

## Convert GeoJSON search results to STAC

Run the script `./create_stac_catalog.py` to setup a catalog with links to the GCS assets, e.g.:

```shell
# Create Sentinel-2 L1C data catalog
python create_stac_catalog.py ./data/sentinel-2/sentinel2-l1c.json --path ./data/sentinel-2/sentinel2-l1c --description "Sentinel-2 L1C scenes of the Brintnell-Bologna icefield" --template "${year}/${month}/${day}"
# Create Sentinel-2 L2A data catalog
python create_stac_catalog.py ./data/sentinel-2/sentinel2-l2a.json --path ./data/sentinel-2/sentinel2-l2a --description "Sentinel-2 L2A scenes of the Brintnell-Bologna icefield" --template "${year}/${month}/${day}"
```

## Download imagery

Run the script `./download_assets.py` to retrieve (a subset of) the catalogs' assets:

```shell
# Download Sentinel-2 L1C data (few bands and metadata files)
python download_assets.py --assets blue green red nir product_metadata granule_metadata inspire_metadata datastrip_metadata sensor-metadata-B02 sensor-metadata-B03 sensor-metadata-B04 sensor-metadata-B08
# Download Sentinel-2 L2A data (only scene classification layer)
python download_assets.py --assets scl
```