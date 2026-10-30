# AntADatabase

**Radar isochrones of Antarctica in one database: [antadatabase.com](https://antadatabase.com)**

- [3D viewer](https://antadatabase.com/viewer/): explore the isochrones in 3D in the browser
- [Python database](https://docs.antadatabase.com/intro.html): the `anta_database` package and its documentation
- [Download the data](https://antadatabase.com/download/)
- [Submit your data](https://antadatabase.com/submit/) (coming soon)

## 3D viewer

[![3D view of Antarctica with radar isochrones coloured by age over the Bedmap3 bed topography](og-image.jpg)](https://antadatabase.com/viewer/)

An interactive 3D map of the internal reflecting horizons (IRHs, or isochrones) traced in
ice-penetrating radar data across Antarctica, from the
[AntADatabase](https://github.com/antoinehermant/anta_database). It runs in the browser, with
nothing to install.

### What you can do

- Rotate, zoom and pan over the Bedmap3 bed and ice surface, with adjustable vertical exaggeration,
  or switch to a top view to read it as a map.
- Colour the isochrones by age, depth or elevation.
- Hover a line to see its dataset, age, depth, line ID and citation.
- Show the radar tracks at the ice surface, with the number of isochrones traced at each point and
  the ages of the isochrones of the dataset.
- Show the MEaSUREs ice velocity, with animated flow lines.
- Locate the deep ice cores, with their drilled depth and the age of their oldest ice.
- Share a view: the camera position is kept in the link.
- Download the data: choose datasets on [antadatabase.com/download](https://antadatabase.com/download/)
  and get them in one zip file.

## Data

- Isochrones: [AntADatabase](https://github.com/antoinehermant/anta_database), compiled from
  published datasets of the AntArchitecture community. Please cite the original datasets (shown
  in the tooltips) when using them.
- Bed and surface topography: [Bedmap3](https://doi.org/10.5285/2d0e4791-8e20-46a3-80e4-f5f6716025d2)
  (Pritchard et al., CC-BY 4.0).
- Ice velocity: [MEaSUREs InSAR-Based Antarctica Ice Velocity Map, v2](https://doi.org/10.5067/D7GK8F5J8M8R)
  (Rignot, Mouginot and Scheuchl, NSIDC-0484).

## About this repository

This repository only hosts the built website of antadatabase.com (GitHub Pages). It is generated, and overwritten on
each release, from the [`web/`](https://github.com/antoinehermant/anta_database/tree/main/web)
directory of the [anta_database](https://github.com/antoinehermant/anta_database) repository,
where the source code lives. Please open issues and suggestions
[there](https://github.com/antoinehermant/anta_database/issues).

The data shown by the viewer is read from the public AntADatabase bucket. To use the data in your
own work, see the Python package: `pip install anta_database`
([documentation](https://docs.antadatabase.com/intro.html)).

Developed as part of the CHARIBDIS project (Swiss National Science Foundation grant no. 211542).
MIT licence.
