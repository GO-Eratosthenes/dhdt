# Brintnell-Bologna Icefield

## Run on Spider Data Processing @ SURF


### 1. Download Sentinel-2 data

This step requires access to the Copernicus Open Access Hub (registration [here][coah]). Credentials can be passed to the script using the `COPERNICUS_HUB_USERNAME` and `COPERNICUS_HUB_PASSWORD` environment variables:

```shell
sbatch --export=PYTHON_SCRIPT="./01-download-sentinel-2-data.py",CONDA_ENV="dhdt",ROOT_DIR="/project/eratosthenes/Data/",COPERNICUS_HUB_USERNAME="<USERNAME>",COPERNICUS_HUB_PASSWORD="<PASSWORD>" run-script-on-spider.bsh
```

[coah]: https://scihub.copernicus.eu/userguide/SelfRegistration


### 2. Download auxiliary datasets

This step requires using ERA5 data, which we access from the Copernicus Climate Data Store (CDS). We set up the account and credentials in the `~/.cdsapirc` file as described [here][cdsapi].

```shell
sbatch --export=PYTHON_SCRIPT="./02-download-auxiliary-datasets.py",CONDA_ENV="dhdt",ROOT_DIR="/project/eratosthenes/Data/" run-script-on-spider.bsh
```

[cdsapi]: https://github.com/ecmwf/cdsapi

### 3. Preprocessing

```shell
# get list of item ids
ITEM_IDS=`ls /project/eratosthenes/Data/data/SEN2/sentinel2-l1c-small/*/*/*/ | grep "MSIL1C"`

# submit preprocessing jobs
for ITEM_ID in ${ITEM_IDS} ; do 
    sbatch --export=PYTHON_SCRIPT="./03-preprocess-imagery.py",CONDA_ENV="dhdt",ROOT_DIR="/project/eratosthenes/Data/",ITEM_ID="${ITEM_ID}" run-script-on-spider.bsh
done
```

### 4. Processing

```shell
# submit processing jobs
for ITEM_ID in ${ITEM_IDS} ; do
    sbatch --export=PYTHON_SCRIPT="./04-process-imagery.py",CONDA_ENV="dhdt",ROOT_DIR="/project/eratosthenes/Data/",ITEM_ID="${ITEM_ID}" run-script-on-spider.bsh
done
```

### 5. Postprocessing

```shell
sbatch --export=PYTHON_SCRIPT="./05-extract-elevation-change.py",CONDA_ENV="dhdt",ROOT_DIR="/project/eratosthenes/Data/" run-script-on-spider.bsh
```
