# Anta Database Jupyter Notebooks

This directory contains Jupyter notebooks demonstrating how to use the Anta Database package.

## Binder Access

Launch these notebooks in Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/your-username/anta_database/main?urlpath=lab/tree/notebooks)

## Available Notebooks

- `cloud_database_example.ipynb`: Example using CloudDatabase with S3 zarr data
- `local_database_example.ipynb`: Example using local Database with HDF5 data
- `advanced_plotting.ipynb`: Advanced plotting examples

## Setup for Local Use

```bash
# Create conda environment
conda env create -f environment.yml
conda activate anta-database-binder

# Launch Jupyter Lab
jupyter lab
```

## CloudDatabase Example

The `cloud_database_example.ipynb` notebook shows how to:
1. Connect to the cloud database on S3
2. Query for specific flight lines
3. Plot transect data using zarr files from S3
4. Generate various visualizations
