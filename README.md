<h1 align="center">
<a href="https://antoinehermant.github.io/anta_database/index">
<img src="https://raw.githubusercontent.com/antoinehermant/anta_database/main/book/logo.png" width="200">
</a>
</h1>

[![PyPI version](https://img.shields.io/pypi/v/anta_database)](https://pypi.org/project/anta_database/)
[![Downloads](https://img.shields.io/pypi/dm/anta_database)](https://pypi.org/project/anta_database/)
[![GitHub issues](https://img.shields.io/badge/issue_tracking-github-blue.svg)](https://github.com/antoinehermant/anta_database/issues)

# AntADatabase

**Visit the [Home Page](https://antoinehermant.github.io/anta_database/index) for full documentation and examples.**

**SQL database for the AntArchitecture Community Data**

AntADatabase is a Python package providing efficient access to Internal Reflecting Horizons (isochrones) data across Antarctica. Designed for ice sheet modelers, it offers fast, memory-efficient data structures to constrain models with published IRH data.

## Key Features

- **SQL Indexing**: Fast querying of IRH data by dataset, institute, project, age, region, basin, variable, or flight ID
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

For the latest development version:
```bash
pip install git+https://github.com/antoinehermant/anta_database.git
```

## Quick Start

```python
from anta_database import Database

# Initialize database (contact maintainer for data access)
db = Database("/path/to/AntADatabase/")

# Query data by various criteria
db.query(dataset="Cavitte_2020", age=10000)

# Visualize data
db.plot.var(title="My IRH Data")
```

## Documentation

Full documentation and examples are available on the [Home Page](https://antoinehermant.github.io/anta_database/index).

## Data Access

The database files are not publicly hosted. Please contact [antoine.hermant@unibe.ch](mailto:antoine.hermant@unibe.ch) to request access to the data.

## Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/antoinehermant/anta_database/issues)
- **Contact**: antoine.hermant@unibe.ch

## Citation

If you use this tool in your work, please cite this repository and the original data sources using their provided DOIs.

## Acknowledgments

Developed as part of the CHARIBDIS project (Swiss National Science Foundation grant no. 211542).

