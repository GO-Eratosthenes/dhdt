# Brintnell-Bologna Icefield

## 1. Download auxiliary datasets

```shell
sbatch --export=PYTHON_SCRIPT="./01-download-auxiliary-datasets.py",CONDA_ENV="dhdt",DATA_DIR="/project/eratosthenes/Data/" run-script-on-spider.bsh
```

## 2. Download Sentinel-2 data

## 3. Preprocessing

```shell
# get list of item ids
ITEM_IDS=`ls /project/eratosthenes/Data/SEN2/sentinel2-l1c-small/*/*/*/ | grep "MSIL1C"`

# submit preprocessing jobs
for ITEM_ID in ${ITEM_IDS} ; do 
    sbatch --export=PYTHON_SCRIPT="./03-preprocess-imagery.py",CONDA_ENV="dhdt",DATA_DIR="/project/eratosthenes/Data/",ITEM_ID="${ITEM_ID}" run-script-on-spider.bsh
done
```
