# AntADatabase

This Python module provides an efficient SQLite database for browsing, visualizing and processing Internal Reflecting Horizons (isochrones) across Antarctica, curated by the AntArchitecture action group. It is specifically designed for ice dynamic modelers who need a fast, memory-efficient data structure to constrain their models.

## SQLite Database

The database uses SQLite for efficient indexing. Data is sourced from the associated DOIs and stored as binary DataFrame files for each layer (IRH) and trace ID. This structure enables:
- Browsing by author (region), layer age, or trace ID.
- Memory efficiency: Only relevant data is read when needed.
- Fast read performance: Lightweight and optimized for speed.

### Datasets currently included:
- Winter et al. 2018, (https://doi.org/10.1594/PANGAEA.895528)
- Cavitte et al. 2020, (https://doi.org/10.15784/601411)
- Beem et al. 2021, (https://doi.org/10.15784/601437)
- Wang et al. 2023, (https://doi.org/10.1594/PANGAEA.958462)
- Mulvaney et al. 2023, (https://doi.pangaea.de/10.1594/PANGAEA.963470)
- Sanderson et al. 2024, (https://doi.org/10.5285/cfafb639-991a-422f-9caa-7793c195d316)
- Franke et al. 2025, (https://doi.org/10.1594/PANGAEA.973266)

<!-- ![alt text](figures/all_data.png) -->
<!-- <div style="display: flex; justify-content: space-around;"> -->
<!--   <img src="figures/all_data.png" width="45%" /> -->
<!--   <img src="figures/AntA_38ka_depth.png" width="45%" /> -->
<!-- ![First Figure](figures/all_data.png){ width=45% } ![Second Figure](figures/AntA_38ka_depth.png){ width=45% } -->
| | |
|:---:|:---:|
| ![First Figure](figures/all_data.png) | ![Second Figure](figures/AntA_38ka_depth.png) |

**Figures** created using plotting functions from this module

## Key Features
- Efficient SQLite indexing
- Quick visualization on Antarctica map
- Generate lazy data for later use

```{tableofcontents}
```
