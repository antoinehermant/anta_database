# AntADatabase Jupyter Notebooks

This directory contains Jupyter notebooks demonstrating how to use the anta_database package.

## Binder Access

Launch these notebooks in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/antoinehermant/anta_database/binder?urlpath=lab/tree/notebooks/)

## Available Notebooks

- `cloud_database.ipynb`: Example using CloudDatabase with S3 zarr data

## Setup for Local Use

```bash
# Create conda environment
conda env create -f environment.yml
conda activate anta-database-binder

# Launch Jupyter Lab
jupyter lab
```

## CloudDatabase Example

The `cloud_database.ipynb` notebook shows how to:
1. Connect to the cloud database on S3
2. Query for specific flight lines
3. Plot transect data using zarr files from S3
4. Generate various visualizations
