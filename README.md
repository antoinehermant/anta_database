<h1 align="center">
<a href="https://docs.antadatabase.com/intro.html">
<img src="https://raw.githubusercontent.com/antoinehermant/anta_database/main/book/logo.png" width="200">
</a>
</h1>

[![PyPI version](https://img.shields.io/pypi/v/anta_database)](https://pypi.org/project/anta_database/)
[![Downloads](https://img.shields.io/pypi/dm/anta_database)](https://pypi.org/project/anta_database/)
[![GitHub issues](https://img.shields.io/badge/issue_tracking-github-blue.svg)](https://github.com/antoinehermant/anta_database/issues)

# AntADatabase

**Visit [antadatabase.com](https://antadatabase.com) to explore the database in 3D or download it, and the [documentation](https://docs.antadatabase.com/intro.html) for full examples.**

**SQL database for the AntArchitecture Community Data**

AntADatabase is a Python package providing efficient access to Internal Reflecting Horizons (isochrones) data across Antarctica. Designed for ice sheet modelers, it offers fast, memory-efficient data structures to constrain models with published IRH data.

## Key Features

- **SQL Indexing**: Fast querying of IRH data by dataset, institute, project, age, region, basin, variable, or line ID
- **HDF5 Storage**: Data stored in HDF5 format for optimal read performance
- **Plotting Tools**: Built-in visualization functions for quick data exploration
- **Lazy Data Generation**: Generate data for later use

## Data Variables

Each dataset contains (when available):
- `PSX`, `PSY`: Coordinates
- `Distance`: Along-track distance
- `IRH_DEPTH`: Internal Reflecting Horizon depth
- `IRH_NUM`: Number of traced IRHs per point
- `ICE_THK`: Ice thickness
- `SURF_ELEV`: Surface elevation
- `BED_ELEV`: Bed elevation

## Installation

```bash
pip install anta_database
```

This installs what is needed to query and plot the database. Optional extras:
```bash
pip install "anta_database[compile]"  # compile the database from raw data
pip install "anta_database[pism]"     # plot comparisons with PISM outputs
pip install "anta_database[all]"      # everything
```

For the latest development version:
```bash
pip install git+https://github.com/antoinehermant/anta_database.git
```

## Downloading the data

Download the database, or update it to the latest published version (only new or changed files are downloaded):
```python
from anta_database import sync
sync()                            # default local directory
sync("/path/to/AntADatabase/")    # or a directory of your choice
```
or from the command line: `anta-sync --dir /path/to/AntADatabase/`. From an existing `Database`, use `db.sync()`, or `Database(sync=True)` to always sync when loading.

When loading a synced database, `Database` tells you if a newer version is published (disable with `check_updates=False`). Updates are never downloaded without you asking, so your results only change when you decide to sync.

Keep the package up to date as well (`pip install -U anta_database`): if some features of the latest data need a newer package, `sync()` and `Database` print a warning. In the unlikely case of a breaking change to the data format, each package version keeps syncing the data it can read, and a newer package is needed to get the new data.

## Quick Start

```python
from anta_database import Database

# Load the database (without a path, uses the default directory of sync())
db = Database()

# Query data by various criteria
db.query(dataset="Cavitte_2020", age=10000)

# Visualize data
db.plot.var(title="My IRH Data")
```

## Documentation

Full documentation and examples are available in the [documentation](https://docs.antadatabase.com/intro.html).

## Data Access

The AntADatabase is free to download with `sync()` (see [Downloading the data](#downloading-the-data)). Any data used must be cited at source using the DOIs provided in the metadata.

## Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/antoinehermant/anta_database/issues)
- **Contact**: antoine.hermant@unibe.ch

## Citation

If you use this tool in your work, please cite this repository and the original data sources using their provided DOIs.

## Acknowledgments

Developed as part of the CHARIBDIS project (Swiss National Science Foundation grant no. 211542).

