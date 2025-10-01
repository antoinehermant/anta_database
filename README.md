<h1 align="center">
<img src="https://raw.githubusercontent.com/antoinehermant/anta_database/main/book/logo.png" width="200">
</h1><br>

[![PyPi](https://img.shields.io/pypi/v/anta_database)](https://pypi.org/project/anta_database/)
[![Conda](https://img.shields.io/conda/vn/conda-forge/anta_database)](https://anaconda.org/conda-forge/anta_database)
[![Downloads](https://img.shields.io/pypi/dm/anta_database)](https://pypi.org/project/anta_database)

[![GitHub issues](https://img.shields.io/badge/issue_tracking-github-blue.svg)](https://github.com/antoinehermant/anta_database/issues)
<!-- [![Contributing](https://img.shields.io/badge/PR-Welcome-%23FF8300.svg?)](https://matplotlib.org/stable/devel/index.html) -->

AntADatabase is an efficient SQLite database for browsing, visualizing and processing Internal Reflecting Horizons (isochrones) across Antarctica, curated by the AntArchitecture action group. It is specifically designed for ice dynamic modelers who need a fast, memory-efficient data structure to constrain their models.

Visit the [Home Page](https://antoinehermant.github.io/anta_database/intro) for more information.

# SQLite Database

The database uses SQLite for efficient indexing. Data is sourced from the associated DOIs and stored as binary DataFrame files for each layer (IRH) and trace ID. This structure enables:

-   Browsing by author (region), layer age, or trace ID.
-   Memory efficiency: Only relevant data is read when needed.
-   Fast read performance: Lightweight and optimized for speed.

# Key Features

-   Efficient SQLite indexing
-   Quick visualization on Antarctica map
-   Generate lazy data for later use

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

# Acknowledgments

I am developing this tool as part of my PhD project, which is funded by the Swiss National Science Foundation (grant no. 211542, Project CHARIBDIS)
Any data used through this database should be cited at source. For this, use the DOI provided in the metadata.
If you used this tool for your work and this was useful, please cite this repo, so other people get to know that it exists.

