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

```
AntADatabase/
├── AntADatabase.db
├── database_index.csv #List of directories to index: Author_YYYY,Author et al. YYYY,doi
├── Author_YYYY
    ├── IRH_ages.tab #IRH file names without .ext followed by there respective age in years
    ├── original_new_column_names.csv #first row: names of columns to keep from raw files, second row: how the columns should be renamed
    ├── raw/ #Directory with the original files to process
    └── pkl/ #Directory were the processed files will be written (it will be created in the process)
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
    './Bodart_2021/',
    './Muldoon_2023/',
    './Ashmore_2020/',
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

## Notes
### Multiprocessing
The compilation uses multiprocessing tools for parallel processing. It first finds all the raw files to process and then distribute the processes on multiple processors. By default, it uses all available cpus on the machine minus 1 (to not completely freeze the machine). However, if there are fewer tasks than cpus (fewer files to process), it will use only as many cpus as there are tasks. 
To manually fix the number of cpus used during the compilation:

```
compiler.compile(cpus=2) # Or any integer of choice
```

### Compilation process
The compilation is divided into 2 processes. First it compiles the database by ordering the data in individual trace and individual age. Then a post compilation process loop through all the new pickle files and currently performs the following:
- If variables such as IceThk, SurfElev and BedElev were not given in a distinct file in the raw directory but are present along the IRH, extract them from the IRH files and save it as a separate file. This allows to treat those variables as layers (convenient for visualization, interpolation etc.) 
- Compute the IRH density (IRHDensity) and save it per trace

This structure makes it easy to add post compilation operations on the database, if one wants to add features to the data. One can then perform the post compilation only without having to recompile the database:

```
compiler = CompileDatabase(dir_path, comp=False, post=True) # They are both True by default
compiler.compile()
```
