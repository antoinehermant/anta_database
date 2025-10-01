# AntADatabase

This Python module provides an efficient SQLite database for browsing, visualizing and processing Internal Reflecting Horizons (isochrones) across Antarctica, curated by the AntArchitecture action group. It is specifically designed for ice dynamic modelers who need a fast, memory-efficient data structure to constrain their models.

## SQLite Database

The database uses SQLite for efficient indexing. Data is sourced from the associated DOIs and stored as binary DataFrame files for each layer (IRH) and trace ID. This structure enables:
- Browsing by author (region), layer age, or trace ID.
- Memory efficiency: Only relevant data is read when needed.
- Fast read performance: Lightweight and optimized for speed.

### Overview
![Figure](figures/Overview_all.png) 
**Figures** created using plotting functions from this module

### Datasets currently included:
- Franke et al. 2025, https://doi.org/10.1594/PANGAEA.973266
- Sanderson et al. 2024, https://doi.org/10.5285/cfafb639-991a-422f-9caa-7793c195d316
- Chung et al. 2023 https://doi.pangaea.de/10.1594/PANGAEA.957176
- Muldoon et al. 2023, https://doi.org/10.15784/601673
- Mulvaney et al. 2023, https://doi.pangaea.de/10.1594/PANGAEA.963470
- Wang et al. 2023, https://doi.org/10.1594/PANGAEA.958462
- Bodart et al. 2021, https://doi.org/10.5285/F2DE31AF-9F83-44F8-9584-F0190A2CC3EB
- Beem et al. 2021, https://doi.org/10.15784/601437
- Ashmore et al. 2020, https://doi.org/10.1029/2019GL086663
- Cavitte et al. 2020, https://doi.org/10.15784/601411
- Winter et al. 2018, https://doi.org/10.1594/PANGAEA.895528
- Leysinger et al. 2011, https://doi.org/10.1029/2010JF001785
- Jacobel et al 2005, https://doi.org/10.7265/N5R20Z9T

## Key Features
- Efficient SQLite indexing
- Quick visualization on Antarctica map
- Generate lazy data for later use

## Acknowledgments

I am developing this tool as part of my PhD project, which is funded by the Swiss National Science Foundation (grant no. 211542, Project CHARIBDIS)
Any data used through this database should be cited at source. For this, use the DOI provided in the metadata.
If you used this tool for your work and this was useful, please cite this module, so other people get to know that it exists.

## Tutorial

To get started with this module and get an overview of the features and capabilities, visit:

```{tableofcontents}
```
