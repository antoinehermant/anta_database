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

# Advanced: Database Management

This section covers advanced topics for users who want to compile their own datasets or manage existing AntADatabase installations.

## Database Structure

The AntADatabase uses a specific directory structure to organize datasets:

```
AntADatabase/
├── AntADatabase.db          # SQLite index database
├── FirstAuthor_YYYY/        # Dataset directory (e.g., Cavitte_2020)
│   ├── raw_md.json          # Metadata file (required)
│   ├── original_new_column_names.csv  # Column mapping file (required)
│   ├── raw/                 # Original raw data files
│   └── h5/                  # Processed HDF5 files (created during compilation)
└── ...
```

## Reindexing the Database

Reindexing updates the SQLite database with current metadata. This is necessary when:
- You've modified the `raw_md.json` file for a dataset
- You've added new datasets
- You've changed age information or other metadata

### Method 1: Using IndexDatabase class
```python
from anta_database import IndexDatabase

# Path to your AntADatabase directory
db_path = '/path/to/AntADatabase/'
indexing = IndexDatabase(db_path)
indexing.index_database()
```

### Method 2: Automatic reindexing during Database initialization
```python
from anta_database import Database

# This will automatically run indexing when initializing
db_path = '/path/to/AntADatabase/'
db = Database(db_path, index_database=True)
```

## Compiling Datasets

Compile raw data files into the standardized HDF5 format used by AntADatabase.

### Prerequisites
1. **Directory Structure**: Create a directory for your dataset following the structure above
2. **Metadata File**: Create a `raw_md.json` file with dataset metadata (see below)
3. **Column Mapping**: Create an `original_new_column_names.csv` file (see below)
4. **Raw Data**: Place your raw data files in the `raw/` directory

### Basic Compilation
```python
from anta_database import CompileDatabase

# List of dataset directories to compile
dir_path_list = [
    './Winter_2018',
    './Sanderson_2024',
    './Franke_2025',
    './Cavitte_2020',
    './Beem_2021',
    './Bodart_2021',
    './Muldoon_2023',
    './Ashmore_2020',
]

compiler = CompileDatabase(dir_path_list)
compiler.compile()
```

### Handling TWT Data
If your depth data is in Two-Way Travel Time (TWT) rather than meters, specify the wave speed and firn correction:

```python
dir_path = './Wang_2023'
compiler = CompileDatabase(
    dir_path, 
    wave_speed=0.1685,    # Wave speed in m/ns (units must match file)
    firn_correction=15.5  # Firn correction in meters
)
compiler.compile()
```

### Multiprocessing
The compilation uses multiprocessing for better performance. By default, it uses all available CPUs minus one. To manually set the number of CPUs:

```python
compiler.compile(cpus=4)  # Use exactly 4 CPUs
```

## Metadata File (raw_md.json)

The `raw_md.json` file contains all metadata for your dataset, enabling the database to be queried with all implemented filters. Provide as much information as possible.

### Dataset Organization Types

The `data type` field specifies how your raw files are organized. There are two main types:

#### Type 1: Layer-based Organization
For datasets where each file contains data for a specific layer/age:

| Field | Description | Example Values |
|-------|-------------|----------------|
| `data type` | Must be set to `"layer"` | `'layer'` |
| `raw file` | Name of the raw file in the raw folder with extension | `'OIA_EDC_IRH1.csv'` |
| `dataset` | Dataset name (usually matches folder name) | `'Cavitte_2020'` |
| `institute` | Institute(s) that produced the data | `'BAS'`, `['AWI', 'NASA']` |
| `project` | Project(s) under which data was collected | `'OIB'` |
| `acquisition year` | Year(s) of radar data acquisition (can be range) | `'2000-2010'`, `'2005'` |
| `age` | Age in years before present of the layer | `'10000'` |
| `age uncertainty` | Uncertainty on the age of the layer | `'800'` |
| `citation` | How the dataset should be cited | `'Cavitte et al. 2020'` |
| `DOI dataset` | DOI linking to original dataset | `'https://doi.org/10.15784/601411'` |
| `DOI publication` | DOI of associated publication | `'https://doi.org/10.5194/essd-13-4759-2021'` |

Example dataset: Cavitte_2020

#### Type 2: Flight Line-based Organization
For datasets where each file contains multiple IRH layers organized by flight lines:

| Field | Description | Example Values |
|-------|-------------|----------------|
| `data type` | Must be set to `"flight line"` | `'flight line'` |
| `IRH name` | Name of the IRH layer as it appears in raw file | `'IRH1'` |
| `age` | Age in years before present of the layer | `'10000'` |

Example dataset: Wang_2023

## Column Mapping File (original_new_column_names.csv)

This CSV file maps columns from your raw data files to the standardized variable names used in AntADatabase.

### Required Variables
At minimum, you must provide:
- Polar Stereographic coordinates (`PSX` and `PSY`) or longitude/latitude (will be converted to PSX, PSY)
- The IRH depth

### Standard Variable Names
When mapping your columns, use these standardized names:

| Your Data | Database Convention |
|-----------|---------------------|
| IRH depth | `IRH_DEPTH` |
| Ice thickness | `ICE_THK` |
| Surface elevation | `SURF_ELEV` |
| Bed elevation | `BED_ELEV` |
| Distance along transect | `DIST` |
| Acquisition year | `acq_year` |
| Flight ID | `Flight_ID` |

### File Format
The CSV file should have:
- **First row**: Original column names from your raw files
- **Second row**: Corresponding standardized variable names
- **Third row (optional)**: Pattern to extract year from string (if acquisition year is embedded in a string)

### Examples

#### Example 1: Basic layer-based dataset
```
timestamp,x,y,thk_m,bedelv_mABSL,IRHdepth_m,line
acq_year,PSX,PSY,ICE_THK,BED_ELEV,IRH_DEPTH,Flight_ID
YYYYxxxxxxxxxxxxxxxxxxxxxx
```

#### Example 2: Flight line-based dataset with multiple IRH layers
```
Longitude,Latitude,TWT [ns] (of bed),TWT [ns] (of H1),TWT [ns] (of H2),TWT [ns] (of H3)
lon,lat,ICE_THK,IRH1,IRH2,IRH3
```

**Note**: For flight line-based datasets, the new column names for the layers must match the names you provided in the JSON metadata file.

## File Formatting Notes
- Header information in raw files should be commented out with `#`
- Column names in the raw files should be preserved in the first row of your mapping CSV


## Additional Notes

### Multiprocessing
The compilation process uses multiprocessing for parallel processing. It automatically:
- Discovers all raw files to process
- Distributes tasks across available processors
- Uses all available CPUs minus one by default (to keep your machine responsive)
- Automatically scales down if there are fewer files than CPUs

To manually control the number of CPUs used:

```python
compiler.compile(cpus=2)  # Use exactly 2 CPUs
```

### Data File Requirements
- Header information in raw files must be commented out with `#`
- Column names should be preserved in the first row of your raw files
- Supported file formats: CSV, TXT, TAB

### After Compilation
After compiling your datasets, remember to:
1. Reindex the database (see [Reindexing the Database](#reindexing-the-database))
2. Verify the data by running some test queries
3. Check the processed HDF5 files in the `h5/` directory 
