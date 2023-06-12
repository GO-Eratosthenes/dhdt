# Brintnell-Bologna Icefield

## Run on Spider Data Processing @ SURF


### 1. Download auxiliary datasets

```shell
sbatch --export=PYTHON_SCRIPT="./01-download-auxiliary-datasets.py",CONDA_ENV="dhdt",DATA_DIR="/project/eratosthenes/Data/" run-script-on-spider.bsh
```

### 2. Download Sentinel-2 data

### 3. Preprocessing

```shell
# get list of item ids
ITEM_IDS=`ls /project/eratosthenes/Data/SEN2/sentinel2-l1c-small/*/*/*/ | grep "MSIL1C"`

# submit preprocessing jobs
for ITEM_ID in ${ITEM_IDS} ; do 
    sbatch --export=PYTHON_SCRIPT="./03-preprocess-imagery.py",CONDA_ENV="dhdt",DATA_DIR="/project/eratosthenes/Data/",ITEM_ID="${ITEM_ID}" run-script-on-spider.bsh
done
```

### 4. Processing

This step requires using ERA5 data, which we access from the Copernicus Climate Data Store (CDS). We set up the account and credentials in the `~/.cdsapirc` file as described [here][cdsapi].

```shell
# submit processing jobs
for ITEM_ID in ${ITEM_IDS} ; do
    sbatch --export=PYTHON_SCRIPT="./04-process-imagery.py",CONDA_ENV="dhdt",DATA_DIR="/project/eratosthenes/Data/",ITEM_ID="${ITEM_ID}" run-script-on-spider.bsh
done
```

.. _cdsapi : https://github.com/ecmwf/cdsapi
