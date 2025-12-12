---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Advanced: Managing the database

## Reindexing

You may want for example to update the age of the layers in a particular dataset. For this, you just need to modify the ages in the file called `raw_md.json` located under a dataset directory. Then, reindex with the `IndexDatabase` class:

```
from anta_database import IndexDatabase

db_path = '/home/anthe/documents/data/isochrones/AntADatabase/' 
indexing = IndexDatabase(db_path)
indexing.index_database() 
```

Or simply use the `index_database=True` argument when initializing the `Database`, which runs the IndexDatabase class itself: 
```
from anta_database import Database

db_path = '/home/anthe/documents/data/isochrones/AntADatabase/' 
db = Database(db_path, index_database=True)
```

## (Re)compile the database

You can (re)compile the database, if for example you modify some data in the raw directories or if you add a dataset. For this, make sure to follow the structure: 

```
AntADatabase/
├── AntADatabase.db
├── FirstAuthor_YYYY
    ├── raw_md.json # file containing information about the dataset, take example on an existing dataset and see the section below
    ├── original_new_column_names.csv # first row: names of columns to keep from raw files, second row: how the columns should be renamed (see section below)
    ├── raw/ #Directory with the original files to process
    └── h5/ # Directory were the processed files will be written (it will be created in the process)
```

Note that the header information in the raw files should be commented out with '#'. The column names of the dataframe should be kept.
Then use the `CompileDatabase class to compile the database:

```
from anta_database import CompileDatabase

dir_path_list = [ # list of the dataset subdirectories to compile
    './Winter_2018',
    './Sanderson_2024',
    './Franke_2025',
    './Cavitte_2020',
    './Beem_2021',
    './Bodart_2021/',
    './Muldoon_2023/',
    './Ashmore_2020/',
]

compiler = CompileDatabase(dir_path_list)
compiler.compile()
```

Then reindex (see above). Furthermore, if the depth is not given in meters but TWT, you should set the wave\_speed (units should match values in the file) for conversion and firn\_correction (meters):

```
dir_path = './Wang_2023'
compiler = CompileDatabase(dir_path, wave_speed=0.1685, firn_correction=15.5)
compiler.compile()
```

## The JSON file

The `raw_md.json` defines all the metadata of a raw file which we be associated to the corresponding data in the database. This will allow to query the database with all the implemented filters. One should provide all the information that exists if possible:
One important field is the `data type`, which specifies if the raw files are organised per layers or per flight lines. There are then two possibilities:

`data type`: "layer"
| field         | Description                                                                                     | Example Values                     |
|-------------------|-------------------------------------------------------------------------------------------------|-------------------------------------|
| `data type`       | `layer`                                                                                                       | `'layer'` |
| `raw file`        | Name of the raw file in the raw folder with the correct extension (csv, txt, tab).               | `'OIA_EDC_IRH1.csv'` |
| `dataset`         | Name of the dataset corresponding dataset (usually same as the folder name)                       | `'Cavitte_2020'` |
| `institute`       | Institute(s) that produced the data.                                                           | `'BAS'`, `['AWI', 'NASA']`            |
| `project`         | Project(s) under which the data were collected.                                                | `'OIB'`                             |
| `acquisition year`| Year(s) in which the radar data were acquired. Can be a range.                                        | `'2000-2010'`, `'2005'` |
| `age`             | Age in years before present of the layer.               | `'10000'`   |
| `age uncertainty` | Uncertainty on the age of the layer                                                         | `'800'` |
| `citation` | How the dataset should be cited                                                                       | `'Cavitte et al. 2020'` |
| `DOI dataset` | DOI that directly links to the original dataset (e.g zenodo, USAP-DC...)                          | `'https://doi.org/10.15784/601411'` |
| `DOI publication` | DOI of the associated publication that published the data                                     | `'https://doi.org/10.5194/essd-13-4759-2021'` |
Example of such a dataset is Cavitte_2020

`data type`: "flight line"
Replace the `raw file` entry by `IRH name` as they are named in the raw files. Then the compilation will use the file name for flight ID.
| field         | Description                                                                                     | Example Values                     |
|-------------------|-------------------------------------------------------------------------------------------------|-------------------------------------|
| `data type`       | `flight line`    | `'flight line'` |
| `IRH name`        | Name of the IRH layer as is it named in the raw file (e.g. column name)                       | `'IRH1'` |
| `age`             | Age in years before present of the layer.                                                | `'10000'`  |
Example of such a dataset is Wang_2023

## The CSV column file

The original_new_column_names.csv pairs the columns in the raw file with associated variables in the database. The minimal information in the file should be:
- Polar Stereographic coordinates PSX and PSY or lon, lat (will be converted to PSX, PSY)
- The IRH depth

More information can be provided such as flight ID, Ice thickness, acquisition year etc. But the columns should be renamed following the database convention:
| variable         | variable name convention in the Database                                                                                     |
|-------------------|-------------------------------------------------------------------------------------------------|
| IRH depth       | `IRH_DEPTH`                                                                     |
| Ice thickness   | `ICE_THK`     |
| Surface elevation       |     `SURF_ELEV`                   |
| Bed elevation       |    `BED_ELEV`                        |
| Distance along transect         |     `DIST`                     |
| `acquisition year`|         `acq_year`                 |
| `Flight ID`             | `Flight_ID`   |

Note if the acquisition year column is not directly a year YYYY but a string that contains the year, add a third row to provide the pattern to extract the acquisition year from the string.

Example of original_new_column_names.csv file:
```
timestamp,x,y,thk_m,bedelv_mABSL,IRHdepth_m,line
acq_year,PSX,PSY,ICE_THK,BED_ELEV,IRH_DEPTH,Flight_ID
YYYYxxxxxxxxxxxxxxxxxxxxxx
```

For a dataset organized by flight lines, so many IRH depth in the raw files, the original_new_column_names.csv should look like:
```
Longitude,Latitude,TWT [ns] (of bed),TWT [ns] (of H1),TWT [ns] (of H2),TWT [ns] (of H3),TWT [ns] (of H4),TWT [ns] (of H5),TWT [ns] (of H6),TWT [ns] (of H7)
lon,lat,ICE_THK,IRH1,IRH2,IRH3,IRH4,IRH5,IRH6,IRH7
```
Where the new column names for the layers correspond to the names you provided in the JSON file.


## Notes
### Multiprocessing
The compilation uses multiprocessing tools for parallel processing. It first finds all the raw files to process and then distribute the processes on multiple processors. By default, it uses all available cpus on the machine minus 1 (to not completely freeze the machine). However, if there are fewer tasks than cpus (fewer files to process), it will use only as many cpus as there are tasks. 
To manually fix the number of cpus used during the compilation:

```
compiler.compile(cpus=2) # Or any integer of choice
```

### Compilation process
Needs to be updated. 
