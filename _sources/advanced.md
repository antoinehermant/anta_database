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

You may want for example to update the age of the layers in a particular dataset. For this, you just need to modify the ages in the file called ‘IRH_ages.tab’ located under a dataset directory. Then, reindex with the IndexDatabase class:

```
from anta_database import IndexDatabase

db_path = '/home/anthe/documents/data/isochrones/AntADatabase/' 
indexing = IndexDatabase(db_path)
indexing.index_database() 
```

## (Re)compile the database

You can (re)compile the database, if for example you modify some data in the raw directories or if you add a dataset. For this, make sure to follow the structure: 

```{seealso}
AntADatabase/
├── AntADatabase.db
├── database_index.csv #List of directories to index: Author_YYYY,Author et al. YYYY,doi
├── Author_YYYY
    ├── IRH_ages.tab #IRH file names without .ext followed by there respective age in years
    ├── original_new_column_names.csv #first row: names of columns to keep from raw files, second row: how the columns should be renamed
    ├── raw/
    └── pkl/
```
Then use the CompileDatabase class to compile the database:

```
from anta_database import CompileDatabase

dir_path_list = [ # list of the dataset subdirectories to compile
    './Winter_2018',
    './Sanderson_2024',
    './Franke_2025',
    './Cavitte_2020',
    './Beem_2021',
]

compiler = CompileDatabase(dir_path_list)
compiler.compile()
```

Then reindex (see above). By default, it assumes that the files in raw/ are sorted by IRH (one file = one layer and multiple traces). If the files are sorted the other way around (one file = one trace and multiple layers), you can set file\_type=’trace’ in CompileDatabase(). Furthermore, if the depth is not given in meters but TWT, you should set the wave\_speed (units should match values in the file) for conversion and firn\_correction (meters):

```
dir_path = './Wang_2023'
compiler = CompileDatabase(dir_path, file_type='trace', wave_speed=0.1685, firn_correction=15.5)
compiler.compile()
```
