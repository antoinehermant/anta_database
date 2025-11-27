<h1 align="center">
<img src="https://raw.githubusercontent.com/antoinehermant/anta_database/main/book/logo.png" width="200">
</h1><br>

[![PyPi](https://img.shields.io/pypi/v/anta_database)](https://pypi.org/project/anta_database/)
[![Downloads](https://img.shields.io/pypi/dm/anta_database)](https://pypi.org/project/anta_database)
[![GitHub issues](https://img.shields.io/badge/issue_tracking-github-blue.svg)](https://github.com/antoinehermant/anta_database/issues)
<!-- [![Contributing](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://matplotlib.org/stable/devel/index.html) -->
<!-- [![Conda](https://img.shields.io/conda/vn/conda-forge/anta_database)](https://anaconda.org/conda-forge/anta_database) -->

AntADatabase is a Python-powered SQLite database designed for browsing, visualizing and processing Internal Reflecting Horizons (isochrones) across Antarctica, curated by the AntArchitecture action group. It is specifically designed for ice dynamic modelers who need a fast, memory-efficient data structure to constrain their models.

Visit the [Home Page](https://antoinehermant.github.io/anta_database/intro) for more information.

# The AntADatabase

The AntADatabase contains all traced and dated Internal Reflective Horizons (isochrones) that have been published at that date. All the data is organized by flight transect for each dataset. The data is encoded in HDF5 (Hierarchical Data Format) which provides convenient data structure and performant read speeds. 
Variables per file (when exist):
- PSX(point)
- PSY(point)
- Distance(point)
- IRH_DEPTH(point, age)
- IRH_NUM(point, age)
- ICE_THK(point)
- SURF_ELEV(point, age)
- BED_ELEV(point, age)

# anta_database Python module

This Python module provides SQL indexing for the AntADatabase, as well as quick plot functions and generate lazy data for later use. It allows to quickly browse through the database, filtering by:
- dataset
- institute
- project
- age
- acquisition_year (year of radar acquisition)
- region
- IMBIE_basin
- var (variable)
- flight id

For examples of queries, plots, generate data (e.g for model comparison), please see [Quick start](https://antoinehermant.github.io/anta_database/quick_start)

# Installation

The Python module can be directly installed from [PyPI](https://pypi.org/project/anta-database/) with:

    pip install anta_database

Note that this module is new and under development, so that the [PyPI](https://pypi.org/project/anta-database/) package may not contain the latest features. For the latest version and development, see the instruction below.
To get started with the anta_database module, see the [Documentation](https://antoinehermant.github.io/anta_database).
Also, you need the actual data to use this module. It is currently not available on any public repository, so please contact me.

## Advanced installation

One can install the latest commit from this GitHub directory with:

    pip install git+https://github.com/antoinehermant/anta_database.git

Or for development, you should clone this repo and install the module in development mode:

    git clone git@github:antoinehermant/anta_database.git
    pip install -e anta_database/

# Support and contact

You can email me for downloading the database: antoine.hermant@unibe.ch

Feel free to raise an issue on the GitHub if you find any bug or if you would like a feature added.

# Contribution

If you like this database and wish to help me develop this module, do not hesitate to contact me. You should then fork the repo, build feature branches and pull request. That would be much appreciated!
Please have also a look at the [Roadmap](https://github.com/antoinehermant/anta_database/blob/main/ROADMAP.md) to check whether some features are already in development or for ideas.


# Acknowledgments

I am developing this tool as part of my PhD project, which is funded by the Swiss National Science Foundation (grant no. 211542, Project CHARIBDIS)
Any data used through this database should be cited at source. For this, use the DOI provided in the metadata.
If you used this tool for your work and this was useful, please cite this repo, so other people get to know that it exists.

