# AntADatabase: brief for the manuscript

Context for writing the journal paper on the AntADatabase, written from the development sessions
of the database (snapshot of **2026-10-31**). The paper is written in another repository; that
session can read this file and the code with `claude --add-dir <path of this repository>`.

**Sources of truth, in order:** the code of this repository, the published bucket (manifest
below), then this brief. The numbers here were computed on 2026-10-31 from the published
database `v2`, version `2026.10.30-1403` (section 4). Recompute them before submission, and again
after any dataset is added (see section 16).

Package version at the snapshot: **0.6.1** (tag of 2026-10-30), main branch at commit `d920ac9`.

---

## 1. Summary in a few sentences

The AntADatabase compiles the published radar isochrones of Antarctica into one uniform,
queryable database. Isochrones are internal reflecting horizons (IRHs), dated layers of the ice
traced in airborne and ground-based radar data. The work spans three linked layers:

1. **A dataset**: 18 published datasets, 956 radar lines, 160 dated IRHs, 9.5 million points
   and 44 million IRH depth samples. They are converted to one HDF5/netCDF format with standard
   variable names and indexed in SQLite.
2. **A Python package** (`anta_database`, PyPI, MIT licence): query by metadata (dataset, age,
   institute, project, radar instrument, acquisition year, region or IMBIE basin, variable, line),
   stream the data lazily, and plot it. It also syncs a local copy from the cloud, and compiles new
   data.
3. **A website, antadatabase.com.** It has a 3D viewer of all the isochrones over Bedmap3, a
   download page, and a submission portal. Through the portal, contributors upload their own
   data; it is checked, compiled automatically and previewed in 3D privately, then added after
   review by the maintainer.

**Target users:** ice-sheet modellers who need isochrones to constrain or evaluate models
(the PISM comparison tools in the package come from that use), and radioglaciologists who want
the existing traced layers in one place. It is developed in the framework of the **SCAR
AntArchitecture** action group (the README says "AntArchitecture Community Data"; the book says
"Maintained by the AntArchitecture action group"). Check the exact wording and status with the
group before writing it in the paper.

**Funding:** PhD project, CHARIBDIS project, Swiss National Science Foundation grant
no. 211542, University of Bern. Maintainer contact: antoine.hermant@unibe.ch.

---

## 2. Motivation (as built into the design)

These are the problems the design answers, each traceable in the code. They are a good
skeleton for the introduction.

- **Format heterogeneity.** Each published dataset has its own layout:
  - one file per layer (PANGAEA, USAP-DC style), or one file per radar line with layers as columns;
  - its own column names and separators (`,` / tab / space), and its own nodata values;
  - depths in metres or in two-way travel time (TWT), with or without firn correction;
  - lon/lat or polar stereographic coordinates.
  - Ice thickness, surface and bed are sometimes given and sometimes not.

  The compile step (section 6) is the normalisation layer.
- **Metadata scattered or missing.** Ages and uncertainties, institutes, projects, radar systems,
  acquisition years and DOIs live in papers and repository pages. `raw_md.json` puts them, per
  file or per IRH, into one machine-readable record. It feeds the SQLite index and therefore the
  filters.
- **Modellers need cross-dataset queries.** Typical questions are "all IRHs of 38 ka with ice
  thickness", or "everything in basin X". These are SQL queries over the metadata, followed by
  lazy loading of only the matching files.
- **Access and reproducibility.** Before Oct 2026, the data was shared by email. Now there is a
  public bucket with a checksummed manifest; `sync()` downloads only new or changed files; each
  publication has a `db_version`; and data schema versions are kept apart, so results only change
  when the user syncs.
- **Discoverability and visual QC.** The 3D viewer shows where the data is and what has been
  traced. It also exposes errors: a wrong IRH number shows up as crossing lines, a wrong firn
  correction as layers deeper than the bed.
- **Growth without a bottleneck.** The submission portal lets data producers add their data in
  the standard format. Checks run automatically, and the maintainer only reviews and approves.

---

## 3. Components and where they live

| Component | Path | Role |
|---|---|---|
| Compile | `anta_database/compiling/compile.py` (`CompileDatabase`) | raw files + `raw_md.json` + column mapping → one HDF5 file per radar line |
| Index | `anta_database/indexing/index_database.py` (`IndexDatabase`) | assigns ages to the h5 files, builds `AntADatabase.db` (SQLite) |
| Query | `anta_database/database/database.py` (`Database`, `MetadataResult`) | `query`, `filter_out`, `get_files`, `data_generator`, `sync` |
| Plot | `anta_database/plotting/plotting.py` (`db.plot.*`), `plotting_pism.py` (`db.plot_PISM.*`) | maps by dataset, institute, line, variable; stratigraphy; PISM mismatch |
| Cloud | `anta_database/cloud/{manifest,publish,sync}.py` | `anta-publish` (maintainer), `sync()` / `anta-sync` (users) |
| Web exports | `anta_database/web/{terrain,irh,outlines,velocity,catalog}.py` | data products of the viewer and the download page |
| Website | `web/` (Vite, TypeScript, three.js, lil-gui, client-zip) | front page, `/viewer/`, `/download/`, `/submit/`, `/contact/`, `/privacy/` |
| Portal | `portal/` (FastAPI, worker, Caddy, Docker Compose) | `api.antadatabase.com`: submissions, checks, previews, contact, statistics |
| Review | `portal/review.py` (`anta-review`) | maintainer's tool: accept, reject, publish, remove |
| Docs | `book/` (Jupyter Book) | docs.antadatabase.com: intro, quick start, PISM example, advanced, contributing |
| Site build | `publish_site.sh` | builds and publishes the website and the docs to GitHub Pages |

Addresses:
- https://antadatabase.com (site);
- https://antadatabase.com/viewer/ (3D viewer);
- https://docs.antadatabase.com (docs);
- https://api.antadatabase.com (portal API);
- https://github.com/antoinehermant/anta_database (code);
- https://pypi.org/project/anta_database/ (package);
- the old address https://3d-antadatabase.com redirects to the viewer.

---

## 4. Content of the database (published v2, `2026.10.30-1403`)

The manifest is at
`https://s3.pub1.infomaniak.cloud/object/v1/AUTH_2030d6d085ec4b16a7c2320f0bb9427d/antadatabase/v2/manifest.json`.
It lists 975 files: 956 h5 files, 18 `raw_md.json` and the SQLite index. They total **630 MB**,
and the format is `schema_version` 2 with `min_package_version` 0.6.0.

### 4.1 Totals

| | |
|---|---|
| Datasets | 18 |
| Radar lines (h5 files) | 956 |
| IRHs (per-dataset layers) | 160, **all dated** (age given for each) |
| (line × IRH) features in the viewer | 8,085, plus 10 basal-unit features |
| Points (along-track positions) | 9,528,122 |
| IRH depth samples (non-missing `IRH_DEPTH` over all points × IRHs) | 44,058,753 |
| Age range | 0.8 ka (Siegert_2004, IRH 1) to 708.5 ± 54 ka (Cavitte_2020, IRH 26) |
| Institutes (normalised names in the index) | 12 |
| Projects | 24 |
| Radar instruments | 10 named (+ unknown) |
| Regions / IMBIE2 basins touched | 2 (EAIS, WAIS) / 13 basins (regions are from the IMBIE2 basins) |

Note: the 160 IRHs are counted *per dataset*. The same isochrone traced in two datasets (e.g. the
38.1 ka layer in Cavitte_2020, Franke_2025, Yan_2025 and Winter_2018) counts several times. The
database does not identify layers across datasets (section 15).

### 4.2 Per dataset

`structure`:
- `IRH` = one file per layer;
- `line` = one file per radar line.

`track km` is the sum over lines of the maximum along-track `Distance`. It overestimates the
surveyed length where a "line" contains jumps (e.g. Leysinger_Vieli_2011: 1 line, 2,906 points,
13,505 km; Siegert_2004: 1 line, 348 points, 794 km), so don't quote it without fixing that
first (section 15).

| Dataset | Citation | Structure | Lines | IRHs | Ages (ka) | Points | IRH samples | MB |
|---|---|---|---|---|---|---|---|---|
| Ashmore_2020 | Ashmore et al. (2020) | IRH | 58 | 3 | 2.6–6.9 | 807,349 | 1,408,828 | 35.8 |
| Beem_2021 | Beem et al. (2021) | IRH | 15 | 8 | 4.7–93.9 | 104,089 | 597,653 | 7.7 |
| Bodart_2021 | Bodart et al. (2021) | IRH | 34 | 4 | 2.6–16.5 | 805,496 | 1,462,219 | 37.2 |
| Bodart_2025a | Bodart and Sutter (2025a) | IRH | 7 | 7 | 4.8–91.0 | 183,204 | 808,605 | 10.5 |
| Bodart_2025b | Bodart and Sutter (2025b) | IRH | 73 | 9 | 6.6–128.4 | 1,247,278 | 2,568,164 | 59.1 |
| Cavitte_2020 | Cavitte et al. (2020) | IRH | 77 | 26 | 10.0–708.5 | 1,463,368 | 16,201,427 | 136.3 |
| Chung_2023 | Chung et al. (2023) | IRH | 12 | 19 | 73.3–473.0 | 52,281 | 991,271 | 8.7 |
| Franke_2025 | Franke et al. (2025) | IRH | 208 | 9 | 4.8–91.0 | 758,762 | 4,623,080 | 60.8 |
| Jacobel_2005 | Jacobel and Welch (2005) | line | 3 | 1 | 17.5 | 148,559 | 68,540 | 5.4 |
| Leysinger_Vieli_2011 | Leysinger-Vieli et al. (2011) | line | 1 | 9 | 17.5–352.4 | 2,906 | 12,214 | 0.2 |
| Muldoon_2023 | Muldoon et al. (2023) | IRH | 107 | 1 | 4.7 | 998,057 | 998,057 | 58.8 |
| Mulvaney_2023 | Mulvaney et al. (2023) | IRH | 144 | 20 | 10.5–414.6 | 248,504 | 4,869,374 | 42.9 |
| Napoleoni_2026 | Napoleoni et al. (2026a,b) | line | 116 | 7 | 2.3–12.4 | 62,172 | 339,842 | 10.1 |
| Sanderson_2024 | Sanderson et al. (2024) | IRH | 20 | 3 | 38.5–161.9 | 560,373 | 1,413,028 | 24.2 |
| Siegert_2004 | Siegert and Payne (2024)? | line | 1 | 5 | 0.8–16.0 | 348 | 973 | 0.0 |
| Wang_2023 | Wang et al. (2023) | line | 22 | 7 | 31.4–169.1 | 518,333 | 3,324,235 | 47.4 |
| Winter_2018 | Winter et al. (2018) | IRH | 27 | 6 | 38.1–161.1 | 816,302 | 2,020,521 | 38.1 |
| Yan_2025 | Yan et al. (2025) | line | 31 | 16 | 6.5–96.5 | 750,741 | 2,350,722 | 45.9 |

### 4.3 Variables per dataset, and metadata

All datasets have `PSX`, `PSY`, `IRH_DEPTH`, `IRH_AGE`, `IRH_AGE_UNC`, `IRH_NUM` and `Distance`.
In addition:

| Dataset | ICE_THK | SURF_ELEV | BED_ELEV | BASAL_UNIT | Institutes | Projects | Radar | Acquisition |
|---|---|---|---|---|---|---|---|---|
| Ashmore_2020 | | | | | BAS | IMAFI | PASIN | 2010 |
| Beem_2021 | ✓ | ✓ | ✓ | | PRIC, UTIG | SPC, PPT | HICARS | 2016, 2017 |
| Bodart_2021 | ✓ | | | | BAS, NASA-CReSIS | BBAS, OIB | PASIN, MCoRDS | 2004, 2016, 2018 |
| Bodart_2025a | ✓ | ✓ | ✓ | | AWI | CHARISO, ANTR, SNACC-DML | AWI-UWB, EMR | 1999–2025 |
| Bodart_2025b | ✓ | ✓ | ✓ | | AWI, BAS, NASA-CReSIS, UTIG-AAD | WISE_ISODYN, OIB, ICECAP, ANTR | PASIN, MCoRDS, HiCARS, EMR | 2005–2019 |
| Cavitte_2020 | ✓ | ✓ | ✓ | | BAS, NASA-CReSIS, UTIG-AAD | ICECAP-OIA, OIB, BE-OI | (missing) | 2009–2016 (one line "none") |
| Chung_2023 | ✓ | | | ✓ | UA, UTIG | BE-OI | HiCARS, LDC-VHF | (missing) |
| Franke_2025 | ✓ | ✓ | ✓ | | AWI | ANTR | EMR, AWI-UWB | (missing) |
| Jacobel_2005 | ✓ | | | | St-Olaf-College | US-ITASE | (missing) | 2001–2002 |
| Leysinger_Vieli_2011 | ✓ | | | | SPRI-NSF-DTU | (missing) | (missing) | 1970–1979 |
| Muldoon_2023 | ✓ | ✓ | ✓ | | UTIG | SOAR-CASERTZ, SOAR-DVD, ATRS, AGASEA, GIMBLE | (missing) | 2004, 2005 |
| Mulvaney_2023 | ✓ | | | | BAS | BE-OI | DELORES | 2016–2018 |
| Napoleoni_2026 | ✓ | ✓ | ✓ | | Durham University | Subglacial Lake CECs Exploration Program | ULUR, CORDS | 2014 |
| Sanderson_2024 | ✓ | | | | BAS | AGAP-NORTH, POLARGAP | PASIN | 2007–2016 |
| Siegert_2004 | | | | | SPRI-NSF-DTU | (missing) | SPRI-NSF-DTU-radar | 1974–1978 |
| Wang_2023 | ✓ | | | | AWI | BE-OI | EMR | 2016 |
| Winter_2018 | | | | | AWI, LDEO | ANTR, AGAP-SOUTH-GAMBIT | EMR, MCoRDS | 1998–2008 |
| Yan_2025 | | | | | PRIC-UTIG-AAD | ICECAP2 | HiCARS | 2016–2019 |

Point counts: 4 datasets have no thickness or surface of their own. For those, the viewer and the
portal checks use Bedmap3 (sections 10, 11). Names are written as stored (e.g. `NASA-CRESIS`,
`HICARS`, `MCORDS` in the index); harmonise the spelling in the paper. The "Projects" and "Radar"
columns show the list as stored, and some datasets store them per line.

### 4.4 DOIs and ages, per dataset (as stored in `raw_md.json` / the h5 files)

| Dataset | DOI dataset | DOI publication |
|---|---|---|
| Ashmore_2020 | 10.5281/zenodo.4945301 | 10.1029/2019GL086663 |
| Beem_2021 | 10.15784/601437 | 10.5194/tc-15-1719-2021 |
| Bodart_2021 | 10.5285/F2DE31AF-9F83-44F8-9584-F0190A2CC3EB | 10.1029/2020JF005927 |
| Bodart_2025a | 10.5281/zenodo.17348976 | 10.5194/egusphere-2025-5381 |
| Bodart_2025b | 10.5281/zenodo.17348094 | 10.5194/egusphere-2025-5381 |
| Cavitte_2020 | 10.15784/601411 | 10.5194/essd-13-4759-2021 |
| Chung_2023 | 10.1594/PANGAEA.957176 | 10.5194/tc-17-3461-2023 |
| Franke_2025 | 10.1594/PANGAEA.973266 | 10.5194/tc-19-1153-2025 |
| Jacobel_2005 | 10.7265/N5R20Z9T | 10.1029/2003GL017210 |
| Leysinger_Vieli_2011 | 10.5281/zenodo.15516203 | 10.1029/2010JF001785 |
| Muldoon_2023 | 10.15784/601673 | 10.1093/climatesystem/dzy004 (check) |
| Mulvaney_2023 | 10.1594/PANGAEA.963470 | 10.5194/tc-17-3461-2023 (same as Chung_2023: check) |
| Napoleoni_2026 | (none) | 10.5194/tc-20-2793-2026 |
| Sanderson_2024 | 10.5285/cfafb639-991a-422f-9caa-7793c195d316 | 10.1017/jog.2024.60 |
| Siegert_2004 | onlinelibrary.wiley.com/doi/10.1002/esp.1238 (a paper, not a dataset: check) | 10.1029/2004GL020290 |
| Wang_2023 | 10.1594/PANGAEA.958462 | 10.5194/tc-17-4297-2023 |
| Winter_2018 | 10.1594/PANGAEA.895528 | 10.5194/essd-11-1069-2019 |
| Yan_2025 | 10.5281/zenodo.14962526 | 10.1017/jog.2025.15 |

Ages per IRH, in ka (± uncertainty; "–" = no uncertainty given):

- **Ashmore_2020:** 2.615±0.305, 4.72±0.28, 6.94±0.31
- **Beem_2021:** 4.7, 10.7, 16.8, 29.1, 37.6, 51.4, 72.5, 93.9 (all –)
- **Bodart_2021:** 2.615±0.305, 4.72±0.28, 6.94±0.31, 16.5±0.79
- **Bodart_2025a:** 4.8±0.6, 7.4±0.8, 25.6±0.1, 38.1±0.1, 52.2±0.1, 73.4±0.9, 91±1.1
- **Bodart_2025b:** 6.6±0.1, 13.1±0.3, 37.7±0.5, 64.1±2.5, 73.1±1.2, 90.8±1.4, 117±1.3, 122.8±1, 128.4±3.6
- **Cavitte_2020:** 10±0.3, 29.4±0.9, 38.1±0.7, 46.4±0.9, 73.4±2.1, 82±1.6, 96.5±1.8, 106.3±1.9,
  121.1±1.8, 127.8±1.8, 160.4±3.9, 166.4±3.6, 200.1±2.6, 220.1±3.3, 254.5±5.5, 277.9±4.9,
  327.3±4.1, 341.5±5.6, 366.5±7.9, 399.5±9.7, 419.8±13.7, 482.7±22.1, 516.8±21.9, 560.2±27.2,
  611±35.4, 708.5±54
- **Chung_2023:** 73.3±2.42, 84.3±2.02, 90±2.16, 96.5±2.3, 113±2.67, 121±2.22, 132±2.97, 160±5.8,
  179±5.51, 203±4.65, 215±4.5, 239±5.47, 243±5.14, 304±10.8, 320±10.5, 336±9.4, 365±16.2,
  396±13.1, 473±28.3 (+ the basal unit)
- **Franke_2025:** 4.8±0.62, 7.2±0.1, 11.1±0.1, 15.4±1.19, 25.6±0.1, 38.1±0.1, 52.2±0.1, 73.4±0.88, 91±1.1
- **Jacobel_2005:** 17.5 (–)
- **Leysinger_Vieli_2011:** 17.532, 36.5, 62.477, 73, 118.342, 129, 170.514, 244.718, 352.358 (all –)
- **Muldoon_2023:** 4.7±0.246
- **Mulvaney_2023:** 10.516±0.485, 30.397±1.452, 39.215±1.112, 47.11±1.402, 62.235±2.443,
  75.16±2.175, 80.007±1.933, 82.646±1.852, 97.667±2.1, 108.315±2.463, 128.542±1.99, 166.112±4.67,
  202.334±3.471, 240.397±3.791, 262.159±6.89, 326.168±6.566, 345.462±10.077, 381.523±12.33,
  395.802±10.045, 414.644±10.518
- **Napoleoni_2026:** 2.26±0.92, 3.7±1.52, 4.72±2.01, 4.89±0.568, 6.94±3.15, 7.6±0.652, 12.4±5.21
- **Sanderson_2024:** 38.5±2.2, 90.4±3.57, 161.9±6.76
- **Siegert_2004** (IRH numbers 1, 7, 10, 11, 12): 0.8, 3.1, 5.6, 6.4, 16 (all –)
- **Wang_2023:** 31.4±3, 36.3±3.6, 43.6±4.9, 75.3±7, 95.5±7.5, 135.6±10.3, 169.1±19
- **Winter_2018:** 38.2±0.6, 48.3±1.2, 90.2±1.6, 161.1±3.5, 38.1±0.7, 74.2±1.7 (IRH 1 and 5 are both ≈38 ka)
- **Yan_2025:** 6.5±0.6, 8.5±0.9, 9.5±0.8, 10±0.8, 12±0.8, 13.5±0.9, 15±1, 18±0.9, 23.5±1.7, 10±0.3,
  29.4±0.9, 38.1±0.7, 46.4±0.9, 73.4±2.1, 82±1.6, 96.5±1.8 (from IRH 10: the ages of Cavitte_2020)

Ages are stored as published, in years before present (1950, per `var_attrs.json`). There is one
override in `assign_ages.json` (`IRH ID` 19008298: 109,192 ± 200 a). Check which dataset it
belongs to, as no published file has `IRH_ID` yet.

**Not in the published database:** BEDMAP1, BEDMAP2 and BEDMAP3 (Lythe, Fretwell and Frémand
et al., 2022, bed picks). They are listed in `book/intro.md`, and `Database(include_BEDMAP=True)`
exists, but they are not in v2. Decide whether the paper mentions them; the docs need correcting
either way.

---

## 5. Data model

### 5.1 Directory of a database

```
AntADatabase/
├── AntADatabase.db                 SQLite index (published)
├── manifest.json                   written by sync(): the published version
├── assign_ages.json                (optional) age overrides by IRH ID
├── var_attrs.json                  (maintainer) units and descriptions of the variables
└── <FirstAuthor_YYYY[a-z]>/
    ├── raw_md.json                 metadata (published)
    ├── original_new_column_names.csv   column mapping (not published)
    ├── raw/                        original files (not published in v2; see raw/ below)
    └── h5/<line>.h5                one file per radar line (published)
```

The dataset names follow `Author_YYYY` with an optional letter (the portal enforces
`^[A-Za-z][A-Za-z0-9_]*_\d{4}[a-z]?$`).

### 5.2 One h5 file = one radar line

It is written with xarray and h5netcdf, so it is netCDF4-compatible. It has:
- **dimensions** `point` and `IRH`;
- coordinates `point`, `IRH` (the IRH number within the dataset), `IRH_AGE` (on `IRH`), and
  `IRH_ID` when IDs are given (the dimension is then swapped to IDs);
- **variables:**
  - `PSX`, `PSY` (m, EPSG:3031);
  - `IRH_DEPTH(point, IRH)` (m below the ice surface);
  - `IRH_AGE_UNC(IRH)`;
  - when available: `ICE_THK`, `SURF_ELEV` (m a.s.l.), `BED_ELEV` (m a.s.l.) and `BASAL_UNIT`
    (m below the surface: the top of the echo-free zone);
  - computed: `Distance` (m, cumulative along-track) and `IRH_NUM` (number of traced IRHs at the
    point).
- **attributes:** `dataset`, `institute`, `project`, `radar instrument`, `acquisition year`,
  `line ID`, `line ID flag` (original / not_provided / project_acq-year_number), `DOI dataset`,
  `DOI publication`, and `IMBIE_basins` (a JSON map of basin to region, from a spatial join with
  IMBIE2 v1.6).
- **compression and chunks:** zlib level 1. Chunks hold 100 points for lines under 1 M points,
  1,000 under 10 M, and 10,000 above; `IRH_DEPTH` is chunked one IRH at a time. This makes
  reading one IRH cheap.
- **units:** the `units`/`long_name`/`description` attributes come from `var_attrs.json`, and are
  only written when `CompileDatabase(var_attrs_json=...)` is given. So some published datasets
  have them (Cavitte_2020, Napoleoni_2026) and others don't (Wang_2023). To fix before the paper
  (section 15).

### 5.3 Line IDs and data schema versions

- **Schema 1** (prefix `v1/`) used "flight ID".
- **Schema 2** (prefix `v2/`, Oct 2026, package ≥ 0.6.0) renamed it **line ID** (h5 attribute
  `line ID`, SQLite `datasets.line_id`), because ground-based radar lines joined the database
  (e.g. Jacobel_2005 US-ITASE traverses, Siegert/Leysinger-Vieli SPRI lines).
- The old names are still read: `Flight_ID` columns, `flight ID` keys in `raw_md.json`, and
  `flight_id=` arguments, which give a FutureWarning.
- `anta_database/indexing/migrate_line_id.py` migrated the files.
- Breaking changes of the layout are published under a new prefix. Each package version keeps
  syncing the data it can read (section 9).

---

## 6. Compilation (`CompileDatabase`)

The inputs per dataset directory are:

1. **`raw/`**: `.csv` (`,`), `.tab` (tab) or `.txt` (space). Lines starting with `#` are comments.
2. **`original_new_column_names.csv`**:
   - row 1: the original column names;
   - row 2: the standard names (`PSX`, `PSY`, `lon`, `lat`, `Line_ID`, `IRH_DEPTH`, `ICE_THK`,
     `SURF_ELEV`, `BED_ELEV`, `BASAL_UNIT`, `DIST`, `acq_year`, IRH numbers);
   - optional row 3: a pattern such as `YYYYxxxx…` to take the acquisition year out of a
     timestamp column.

   Columns not in the mapping are dropped.
3. **`raw_md.json`**, one record per raw file (structure `IRH`) or per IRH (structure `line`).
   - Keys: `file structure`, `dataset`, `IRH`, `age`, `age uncertainty`, `institute`, `project`,
     `acquisition year` (an integer, or `"YYYY-YYYY"`), `citation`, `DOI dataset`,
     `DOI publication`.
   - Optional: `radar instrument` (a string or a list); for `IRH` only, `raw file`, `line ID`
     (one ID for the whole file) and `line ID prefix`.
   - Institute, project and radar instrument may be lists.

What `compile()` does (multiprocessing pool, all CPUs minus one by default):

1. **Extract**, one task per raw file:
   - read all columns as strings, keep the mapped ones, and rename them;
   - **nodata:** `-9999`, `-9999.0`, `NaN`, `nan` and empty become NA. Rows where both PSX and
     PSY are nodata are dropped, with a warning;
   - convert the numeric columns;
   - **derive the missing third variable** from the other two: `BED_ELEV = SURF_ELEV - ICE_THK`,
     `SURF_ELEV = BED_ELEV + ICE_THK`, `ICE_THK = SURF_ELEV - BED_ELEV`;
   - **TWT:** optionally multiply by `wave_speed` (m/ns, e.g. 0.1685) and add `firn_correction`
     (m, e.g. 15.5), per compile call. This is applied to `IRH_DEPTH`, and, as the code stands, to
     `ICE_THK` and `BED_ELEV` too (section 15);
   - **coordinates:** if PSX/PSY are absent, lon/lat (WGS84) are projected with
     `+proj=stere +lon_0=0 +lat_0=-90 +lat_ts=-71 +datum=WGS84`, i.e. EPSG:3031;
   - **line IDs:** `/` becomes `_`. The per-file `line ID` or `line ID prefix` is applied. A
     missing ID becomes `<project>_<year>`;
   - split the data by line and write `pkl/<line>/<IRH>.pkl` with the line metadata.
   - **`acq_year` column with a pattern:** the year comes from the timestamps, and a line with
     two different years raises an error.
2. **Combine**, one task per line:
   - outer-join the IRHs of the line on (PSX, PSY), dropping duplicate positions (first kept);
   - merge `ICE_THK`, `SURF_ELEV`, `BED_ELEV` and `BASAL_UNIT` coming from several layer files
     by backfill (first non-missing value);
   - sort the IRH columns numerically;
   - write `h5/<line>.h5` (or `<institute>_<line>.h5` when there is a single institute).
3. **Post-processing**, in parallel per file, each step optional:
   - **`Distance`** (cumulative Euclidean distance in PSX/PSY, in file order);
   - **`IRH_NUM`**;
   - **IMBIE2 basins** (point-in-polygon with geopandas; the "Islands" region is excluded);
   - **attributes** (`var_attrs.json`);
   - optionally: transect breaking at jumps over 50 km (for lines with no IDs), and shapefile and
     GeoPackage export (e.g. for QAntarctica).
4. Remove the temporary pickles.

The dependencies of compiling are an optional extra (`pip install "anta_database[compile]"`:
dask, geopandas, pyproj, scipy, netcdf4, zarr). This keeps the user install light (numpy, pandas,
pyarrow, xarray, h5netcdf, h5py, matplotlib, colormaps, tqdm, shapely).

---

## 7. Ages and the SQLite index (`IndexDatabase`)

- Ages are **not written at compile time**. `IndexDatabase(assign_ages=True)` writes them into
  the h5 files from `raw_md.json`, overridden by `assign_ages.json` (by `IRH ID`). Changing an
  age therefore means editing the metadata and re-indexing, with no recompilation.
  - The comparison is NaN-aware and dtype-independent.
  - **A file is only rewritten when its ages change.** This keeps its bytes and checksum, so
    users don't re-download unchanged files.
  - Changes are reported per dataset, e.g. `IRH 3: 38.1 ± 0.7 ka -> 38.4 ± 0.7 ka`.
  - It raises an error when an IRH has two different ages within a dataset.
- **The index is built in a temporary file and then renamed,** so a failure never leaves a broken
  database. Files and directories are sorted, so the same data gives the same database. The index
  is only replaced when its SQL dump changes (`_same_content`), so republishing without changes
  keeps the checksum.
- **Schema:**
  - `sources` (dataset name, citation, DOIs);
  - `datasets` (one row per h5 file: path, source, acquisition year, line ID);
  - dictionary tables: `variables`, `ages(age, age_unc)`, `regions(region, IMBIE_basin)`,
    `projects`, `institutes`, `radar_instruments`;
  - many-to-many link tables `dataset_<x>`.

---

## 8. User API (Python)

```python
from anta_database import Database, sync
sync()                                   # or anta-sync --dir ...
db = Database()                          # default dir; Database(path), Database(sync=True)
md = db.query(age="<50000", var="ICE_THK", region="EAIS", institute="AWI")
db.filter_out(dataset="Winter_2018")     # exclusions
for df, meta in db.data_generator(md, downsampling_factor=10, fraction_depth=True):
    ...                                  # one pandas DataFrame per file and IRH, with its metadata
db.plot.dataset(md); db.plot.var(md); db.plot.stratigraphy(md, elevation=True)
db.plot_PISM.mismatch_2D(...)            # comparison with PISM outputs ([pism] extra)
```

- **`query` filters:** `age`, `var`, `dataset`, `institute`, `project`, `acquisition_year`,
  `line_id`, `region`, `IMBIE_basin` and `radar_instrument`.
  - A value can be exact, a list, a `%` wildcard, a comparison (`>`, `<`, `>=`, `<=`), or a
    year range `YYYY-YYYY`.
  - The result is a `MetadataResult` (citations, DOIs and the matched lines). It is retained for
    `get_files` / `data_generator` / plotting.
- **`data_generator`** loads lazily, one file at a time. Options: `downsampling_factor`,
  `rolling_distance` (along-track smoothing), `fraction_depth`.
- **The default directory** is the platform data directory, e.g.
  `~/.local/share/anta_database/AntADatabase`, or `$ANTA_DATABASE_DIR`.
- **Updates:** `Database` reports when a newer published version exists. It only fetches the
  small manifest, is silent offline, and can be turned off with `check_updates=False`. It never
  downloads without being asked.
- **Docs:** `book/quick_start.ipynb` and `book/pism_example.ipynb`, rendered at
  docs.antadatabase.com.

---

## 9. Distribution: bucket, manifest, versions

- **Storage:** Infomaniak Public Cloud object storage (Swift, with an S3-compatible API), hosted
  in Switzerland, public read, with CORS open. Anonymous reads use the Swift URL.
- **Bucket `antadatabase`:**
  - `v1/`, `v2/`: one prefix per data schema; each holds `manifest.json`, `AntADatabase.db`,
    `*/raw_md.json` and `*/h5/*.h5`;
  - `web/`: viewer and download page data;
  - `raw/<Dataset>/`: the raw files of portal submissions (section 11.7);
  - `test/…`: test publications.
- **Manifest:**
  - `format`, `schema_version`, `min_package_version`, `db_version` (UTC date-time, e.g.
    `2026.10.30-1403`, or a chosen name), `created`;
  - `files` (sha256 and size per path), `datasets` (citation per dataset).
- **`anta-publish`** (maintainer, `[publish]` extra, boto3):
  - uploads only changed files, and the manifest **last**, so readers never see a half-published
    state;
  - deletes only keys listed in the previous manifest;
  - keeps the previous `db_version` when no data file changed.
- **`sync()`:**
  - fetches the manifest and compares it with the local one (falling back to hashing);
  - downloads each new or changed file to `*.part`, checks its sha256, then moves it into place
    (`os.replace`);
  - replaces the SQLite file last, so an interrupted sync leaves a usable database;
  - removes files that left the manifest, and says which datasets it adds, updates or removes;
  - checks the schema and warns when the package is older than `min_package_version`.
- **Download statistics:** after a sync that downloaded files, one anonymous count is sent to the
  portal: first download or later sync, number of files and bytes, package and database versions.
  The portal forwards it to GoatCounter, which keeps the country only and no IP address. It can be
  turned off with `ANTA_NO_STATS=1`, `sync(stats=False)` or `--no-stats`. Website visits are
  counted with GoatCounter too, without cookies. See antadatabase.com/privacy/.
- **History:** v1 was published on 2026-10-29 (`2026.10.29-0642`), and v2 on 2026-10-30
  (`2026.10.30-1403`). The full database is about 630 MB.

---

## 10. Website and 3D viewer (`web/`)

Static site on GitHub Pages: a TypeScript build with Vite, **three.js** rendering and a **lil-gui**
settings panel. The data is read from the bucket. Light and dark themes are available.

### 10.1 Terrain (`anta_database/web/terrain.py`, `web/src/terrain.ts`, shaders)

- **Source:** Bedmap3 (Pritchard et al., 2025, CC-BY 4.0), 500 m, 13,334², EPSG:3031: `topg`,
  `usurf`, `thk` and `mask`.
- **Export:** NaN-aware block means at 2, 4 and 8 km. Mask: the dominant class per block.
  Written as int16 little-endian heightmaps, gzipped, decompressed in the browser with
  `DecompressionStream`; about 25 MB in all.
- **Rendering:**
  - the 8 km preview loads first, then a 4 km mesh;
  - heights are set in the vertex shader, so the **vertical exaggeration** (1–100×) is a single
    uniform, with no re-meshing;
  - lighting (Lambert with a hillshade look) is computed per pixel from the 2 km texture, so the
    relief looks about 2 km sharp while only a 4 km mesh is drawn;
  - the ice surface is a semi-transparent layer with adjustable opacity;
  - frames are only rendered on demand, and resolution is lowered while moving;
  - quality presets are available.
- **Units:** scene units are km in EPSG:3031, and elevation is in m × exaggeration.

### 10.2 Isochrones (`anta_database/web/irh.py`, `web/src/lines.ts`, `irh.ts`)

- **3D position:** `IRH_ELEV = SURF_ELEV - IRH_DEPTH`, with the dataset's own surface where it
  has one, and the Bedmap3 surface otherwise. The surface used is recorded per feature (`surface`,
  `thickness`).
- **Fractional depth** is `IRH_DEPTH / ICE_THK`, with the Bedmap3 thickness where `ICE_THK` is
  missing.
- **Features and runs:**
  - one feature per (line, IRH): 8,085 features, plus 10 basal-unit features;
  - each feature is split into runs at gaps and long jumps, and averaged along-track;
  - each run is stored as float32 vertices (x km, y km, elevation m), with uint16 depth and
    fractional depth;
  - one binary file per dataset (`irh.json` plus the `.bin.gz` files).
- **Surface tracks**, one per line (956): where IRHs are traced, with `IRH_NUM` and the oldest age
  at each point.
- **Profiles**, one per line, fetched on click, for the **stratigraphy plot**:
  - along-track distance, surface, bed, thickness, and the depth of each layer;
  - at full resolution when under `PROFILE_MAX_BYTES` (almost all lines), otherwise binned.
- **Interaction:**
  - colour by age (ka), depth, fractional depth or elevation, with a legend;
  - hover tooltips with the dataset, age ± uncertainty, line, depth, institute and DOI links;
    long press does the same on touch screens;
  - clicking flies to the line and opens its profile; the lines in the way of the open profile
    fade;
  - filters by dataset and by age, applied with a per-segment flag in the vertex shader, so the
    geometry is never rebuilt.
  - The basal unit is drawn as a layer without age.
- **Other layers:**
  - **MEaSUREs ice velocity** (Rignot et al., NSIDC-0484 v2, 450 m → 2 km speed and 4 km
    components), with animated flow lines;
  - **RAISED grounding lines** at 20, 15, 10 and 5 ka (Bentley et al., 2014: confidence classes,
    scenarios A/B) and the present grounding line traced on the Bedmap3 mask;
  - **IMBIE2 ice divides** as walls from the bed to the surface;
  - **deep ice-core sites** (EDC, Beyond EPICA Little Dome C, Dome Fuji, Vostok, EDML, …: position,
    depth, bottom age, reference);
  - animated **survey aircraft** flying the tracks and sending radar pulses, as an illustration
    of how the data is acquired;
  - hidden easter eggs (not for the paper).
- **Usability:**
  - orbit/pan/zoom to the cursor, rotation around the ice under the screen centre, keyboard
    keys and a move pad;
  - the view is kept in the URL (`#view=…`), so views can be shared;
  - the settings of the last visit are remembered;
  - a guided **tour** on the first visit.
- The settings panel has the folders View, Layers, Isochrones, Filter, Light and Performance.

### 10.3 Download page (`/download/`, `web/src/download.ts`, `anta_database/web/catalog.py`)

- A catalogue of the datasets, with their citation, DOIs, institutes, projects, radar, years,
  regions, IRHs and ages, variables, counts and a small coverage map.
- **The file list comes from the published manifest**, so a browser download always matches
  `sync()`.
- The chosen datasets are zipped **in the browser** while they download (client-zip, no
  compression), and streamed straight to disk in Chromium browsers.
  - The whole database comes with the index and the manifest.
  - A selection is re-indexed locally with `Database(path, index_database=True)`, as the
    `README.txt` in the zip explains.
- The page recommends the Python package (`sync()`) first.

### 10.4 Other pages

- the front page;
- `/contact/`: a form through the portal, with verified email and attachments up to 10 files /
  500 MB, stored, not emailed;
- `/privacy/`;
- `/submit/` (section 11);
- `/preview-ended/`.

The site has structured data for search engines, a sitemap and link previews.

---

## 11. Submission portal (`/submit/`, `portal/`)

**Status at the snapshot:** complete, deployed at api.antadatabase.com and tested end to end
with fake submissions (including `anta-review accept --test`). The public page still shows the
"under construction" placeholder, and the form is at `/submit/?dev` until the site is built with
`PORTAL_PUBLIC=true ./publish_site.sh`. **Check before writing that the portal is open.**

### 11.1 The form (all in the browser; a draft is kept in localStorage)

1. **File structure:** one file per layer (with a `LINE_ID` column), or one file per radar line
   (the file name is the line ID, with layer columns `IRH1`, `IRH2`, …). Small diagrams show each.
2. **Publication:** DOI of the publication, with "Fill from DOI"; DOI or URL of the dataset
   (optional but encouraged); first author, year and "et al.". The dataset name `Author_YYYY` is
   suggested and checked against the catalogue.
3. **Survey defaults:** institute(s), project(s), radar instrument(s), and the acquisition year or
   a period `2010-2011`. Names are autocompleted from the catalogue, to keep the vocabularies
   consistent. Files can override these defaults.
4. **Files:** `.csv` or `.tab`. The content must match the extension. Limits: 100 files, 1 GB per
   file, 5 GB per submission. Files are read in pieces: a 1 GB file takes about 25 s to check.
   **Standard column names are required** (there is no mapping step):
   - **coordinates:** `PSX`, `PSY` (EPSG:3031) or `LON`, `LAT` (WGS84);
   - **per layer:** `LINE_ID`, `IRH_DEPTH`. **Per line:** `IRH<n>`;
   - **variables:** `ICE_THK`, `SURF_ELEV`, `BED_ELEV` and `BASAL_UNIT`, as columns of the layer
     files or as separate files with the coordinates and `LINE_ID`, but not both. A file should
     have at least 2 of the 3 reference variables (`ICE_THK`, `SURF_ELEV`, `BED_ELEV`). Otherwise
     there is a warning: "If not provided, the database will try to reconnect the ice thickness
     and bed elevation from Bedmap, but this is not guaranteed". Submitting then asks for
     confirmation;
   - **reserved optional columns,** kept in the raw files but not compiled yet:
     - per layer: `IRH_DEPTH_UNC` (m), `IRH_AGE`, `IRH_AGE_UNC` (a), `IRH_TWT` (µs),
       `IRH_QUALITY` and `IRH_POWER` (dB); in line files, `IRH<n>_DEPTH_UNC` etc.;
     - per point: `SURF_TWT`, `BED_TWT`, `TRACE_NUM`, `DIST` and `TIME` (ISO 8601);
   - unknown columns are accepted with a note asking the contributor to describe them in the
     notes;
   - older names are still accepted: `Line_ID`, `Flight_ID`, `lon`, `lat`.

   Checks in the browser: separator, header, numbers, coordinates in Antarctica (polar
   stereographic extent 3,400 km, or south of 60°S), depths between 0 and 5,000 m, nodata values,
   and a preview of the rows.
5. **Ages:** one row per IRH (age and uncertainty in years, as published), with an optional
   per-file override of the survey defaults. A layer may be left undated.
6. **About you:** name, affiliation, email (verified with a 6-digit code), optional ORCID, and
   notes (wave speed and firn correction used, processing, non-standard columns). The notes are
   published with the raw files.
   - **Consent, three choices:**
     - **"Yes"**: the data is published or about to be, the contributor may share it, and the
       AntADatabase may redistribute it openly with the citation and DOIs, both the compiled data
       and the files as sent;
     - **"Review only"**: the data is checked against the standards of the database but stays
       private and is not added;
     - **"No"**: the submission is blocked.
7. **Check and submit:** a summary and a map of the points.
   - The receipt gives the submission id and the link to a **private 3D preview**.

### 11.2 Backend (FastAPI, `portal/app.py`)

- It validates the payload with pydantic. Accepted: file names `^[A-Za-z0-9][A-Za-z0-9._-]*\.(csv|tab)$`,
  dataset names (pattern above), DOIs, years (`YYYY` or `YYYY-YYYY`).
- It writes the submission as a **dataset directory ready for `CompileDatabase`**:
  `<Dataset>/raw/*`, `raw_md.json` from the form, and `original_new_column_names.csv` as an
  identity mapping (plus `IRHn` → n, and `LINE_ID`/`LON`/`LAT` → the internal names).
  `submission.json` holds the form, the contributor, the purpose and the status.
- Storage: files go to private buckets (`antadatabase-submissions`, `antadatabase-contact`), and
  the server only keeps the small records.
- **Against spam:**
  - email verification (code valid 30 min, 5 tries; the verification holds 24 h);
  - rate limits per address and IP (10 submissions/day);
  - a honeypot field;
  - size limits;
  - a `CLOSED` switch that pauses the portal with a message, without restart.

  ORCID login is a possible later replacement.

### 11.3 Worker pipeline (`portal/worker.py`, `pipeline.py`)

Submissions are processed one at a time. Each runs in its own process with a time limit, in a
container capped at 1.4 GB of memory. A failure only fails that submission, and a job interrupted
by a stop is retried twice at most.

1. `checks.check_raw` re-checks the raw files with the rules of the form, since the API can be
   called without it. **Errors stop here.**
2. `CompileDatabase`, then `IndexDatabase(assign_ages=True)`.
3. `checks.check_compiled` gives **warnings for the reviewer**:
   - points lost by compilation (> 0.5 %);
   - IRH deeper than the ice thickness (`depth > 1.02 × thickness + 10 m`; the dataset's thickness,
     or Bedmap3), suggesting a depth or firn-correction problem;
   - **crossing IRHs**: an older IRH shallower than a younger one by > 1 m at > 1 % of the shared
     points and > 10 points. The IRH crossing the most others is reported first, as a likely wrong
     number or age;
   - **spikes**: depth jumps > 150 m between points < 500 m apart, at > 0.5 % of points;
   - points off the ice of Bedmap3.

   It also gives a summary (lines, IRHs, points, bbox, undated IRHs).
4. The dataset is exported for the viewer (`web.irh`) and the catalogue into a private preview,
   served only with the submission key (`/viewer/?preview=<id>&key=<key>`).
5. The compiled dataset and `report.json` are stored in `antadatabase-compiled` (to publish) or
   `antadatabase-review` (review only).

The maintainer is emailed the report. The contributor follows the progress on the preview page,
and is only emailed when the compilation failed.

### 11.4 The private preview

- The 3D viewer in preview mode shows only the submitted dataset over the same Bedmap3 terrain,
  and frames it once ready.
- It shows the progress of the steps, then the warnings of the checks.
- The contributor can **download their compiled dataset**: h5 files, `raw_md.json` and the
  index.
- It hides what isn't relevant (the dataset filter, aircraft, easter eggs), offers a tour on the
  first visit, and places the stratigraphy profile below the preview panels.
- It is removed when the review ends; the link then shows "this preview has ended".

This is a QC loop for data producers: they see their layers in 3D, in their glaciological
context, before the data is published.

### 11.5 Review (`anta-review`, on the maintainer's computer)

The server holds **no keys of the public bucket**, so a break-in can't alter the published
database. Publishing happens only from the maintainer's machine, from a dedicated **release
copy** that is made identical to the bucket before each change: a sync, then a checksum of
every file.

The commands are `list`, `show`, `pull` (the compiled dataset plus the raw files, to inspect in
Python), `accept`, `reject`, `done` (end a review-only submission: its data is deleted, as
promised), `purge`, `retry`, `delete` (test submissions) and `remove <Dataset>`.

**`accept`:**
1. syncs and verifies the release copy;
2. adds the dataset and rebuilds the index (the ages are already in the files);
3. exports the viewer and catalogue data;
4. shows the plan and asks before publishing;
5. publishes `v2/` and `web/`, and checks them online (the manifest, the viewer index and the
   catalogue);
6. publishes the raw files to `raw/<Dataset>/`;
7. closes the submission on the server (preview and compiled copy removed, raw files kept) and
   emails the contributor;
8. adds the dataset to `book/intro.md` and `book/changelog.md`.

It can resume after a failure. `--replace` publishes a corrected dataset, and `--db-version`
sets the version name. `--test` publishes to `test/v2`, `test/web` and `test/raw` instead,
leaving the submission and the docs alone: the real `v2/` is only read.

**`remove <Dataset>`** is the reverse. It publishes a new version without the dataset, deletes
its raw files and marks the submission `withdrawn`. Users' next `sync()` removes it locally, and
says so.

### 11.6 Infrastructure

- **Server:** one small Infomaniak Public Cloud instance (1 vCPU, 2 GB RAM + 2 GB swap, 30 GB,
  Ubuntu 24.04), in its own cloud project separate from the public data.
  - Firewall: SSH, HTTP and HTTPS only; key-only SSH.
  - Nightly security updates, with reboots when needed (cloud-init).
- **Docker Compose:** portal (FastAPI/uvicorn), worker (with `anta_database[compile]`), and
  **Caddy** for HTTPS with automatic certificates.
- **Private buckets:** `antadatabase-submissions`, `-contact`, `-compiled` and `-review`.
- **Email:** SMTP for the codes, receipts and maintainer notices.
- **Deployment:** `portal/deploy.sh` (rsync and docker compose).
- The checks and previews use a small Bedmap3 extract (`usurf`, `thk`; 84 MB).

### 11.7 Raw archive and provenance

For each accepted submission, `raw/<Dataset>/` in the public bucket holds:
- `raw/`, the files exactly as sent, with **all their columns**, including the optional and
  unknown ones that the compilation doesn't use yet;
- `raw_md.json` and `original_new_column_names.csv`, to recompile them;
- **`provenance.json`**: dataset, citation, DOIs, submission id, date received, database version,
  file structure, the files (kind, IRH, variable, columns, rows, acquisition year, project), the
  layers, the survey and the notes. It does **not** include the contributor's name or email.

There is one copy whatever the data schema, outside the versioned prefixes and in no manifest,
so `sync()` never fetches it. Its purpose is to let later database versions use more columns
(e.g. depth uncertainties) without asking contributors again. Pushing the raw files of the
pre-portal datasets there, after cleaning, is planned.

---

## 12. Engineering, history and licence

- **Code:** Python ≥ 3.11 package, MIT licence, built with setuptools-scm, and published to PyPI
  by GitHub Actions on version tags (trusted publishing). The docs are a Jupyter Book, deployed
  by `publish_site.sh` (and a workflow on changes in `book/`).
- **Size, roughly:** Python package 7,500 lines; web front end 11,600 lines of TS/CSS/HTML/GLSL;
  portal 2,700 lines.
- **Timeline:**
  - first commit 2025-08-27; first PyPI tag 0.1.2 on 2025-09-24;
  - 0.2 (Oct 2025), 0.3.x (Nov–Dec 2025), 0.4.x (Aug 2026);
  - **0.5.0 on 2026-10-27**: cloud publishing and sync;
  - **0.6.0 on 2026-10-30**: line IDs, schema 2;
  - 0.6.1 on 2026-10-30.
  - The viewer, the site and the portal were developed in Oct 2026 (git history).
- **Optional extras:** `[compile]`, `[pism]`, `[publish]`, `[all]`. `CompileDatabase` and
  `PISMPlotting` are imported lazily, with an install hint.
- **Tests:** there is no automated test suite in CI yet. The checks were done by hand:
  - compile, index and query round trips;
  - the fake submissions from `portal/dev/make_fake_data.py` (Chung_2023 turned back into raw
    files in 3 layouts plus a broken set, each giving back the same 12 lines, 19 IRHs and 52,281
    points);
  - publish/sync against test prefixes.

  Reviewers of a software or data paper may ask about tests (section 15).

---

## 13. Design decisions and their reasons (useful for the discussion)

1. **One HDF5 file per radar line, IRH as a dimension.**
   - It matches how the data was acquired, and keeps all layers of a line together for
     stratigraphy.
   - Reading one IRH is cheap thanks to the chunks.
   - Files are netCDF4-compatible, so they are readable without the package.
2. **SQLite for the metadata, HDF5 for the data.** Queries never open data files. The index is a
   single file, published with the data, so users don't need to re-index.
3. **Ages are assigned at indexing, not at compilation.** Re-dating a layer (new ice-core
   chronology) is a metadata edit plus a re-index, and only the changed files change.
4. **Ages are kept as published.** There is no re-dating or harmonisation across datasets, and
   the provenance of each age is the source publication. Harmonising is listed as a future step.
5. **Content-addressed publishing:**
   - a sha256 manifest, uploaded last;
   - downloads to a temporary file, then an atomic move;
   - the SQLite index replaced last.

   This gives incremental updates, consistency during publishing and after an interrupted sync,
   and integrity checks.
6. **Explicit versioning:** `db_version` per publication, a schema prefix per breaking change,
   and `min_package_version`. Users choose when their results change (no silent updates). Old
   packages keep working with old schemas.
7. **Standard column names in the portal instead of a column-mapping UI.**
   - It is simpler and explicit, and the contributor knows their columns.
   - The raw files keep everything.
   - TWT is not converted by the portal: contributors give depths, and the notes record the wave
     speed and firn correction used.
8. **Automatic checks, human decision.** The portal never publishes; the maintainer does, from
   a machine the server can't reach. This separates the attack surface from the public data.
9. **Private 3D preview as feedback to the producers.** It catches numbering, age and depth
   errors before review.
10. **Raw files archived with provenance.** This makes later reprocessing and new variables
    possible, and keeps the contributor's original data citable and traceable.
11. **Lightweight static website.** No server is needed for viewing or downloading: GitHub Pages
    plus the object storage. Only submissions need a server.
12. **Bedmap3 as the fallback reference** for the elevation and fractional depth when a dataset
    doesn't give its own surface or thickness. Each viewer feature records which one was used.

---

## 14. Glossary (wording used in the code and the site)

- **IRH**: internal reflecting horizon, a continuous radar reflector assumed isochronous. Here,
  the "IRH number" is the index of a layer *within its dataset*.
- **Line / radar line**: one survey line (airborne flight line or ground-based traverse). Formerly
  "flight".
- **Point**: one along-track position of a line.
- **Basal unit**: the top of the echo-free zone or basal unit, stored as a depth (`BASAL_UNIT`),
  and shown as an undated layer.
- **Fractional depth**: `IRH_DEPTH / ICE_THK`.
- **IRH_NUM**: number of IRHs traced at a point (the "density" of the stratigraphy).
- **Release copy**: the maintainer's local mirror of the published database, used only for
  publishing.

---

## 15. Caveats, open issues, and what to fix or verify before submission

**Data and metadata (to fix or verify):**
- **DOIs to check:**
  - Mulvaney_2023 has the same publication DOI as Chung_2023 (10.5194/tc-17-3461-2023);
  - Muldoon_2023 publication `10.1093/climatesystem/dzy004`;
  - Siegert_2004's "DOI dataset" is a Wiley paper (esp.1238), its citation reads "Siegert and
    Payne (2024)" while the directory says 2004, and `book/intro.md` says 2024;
  - Napoleoni_2026 has no dataset DOI.
- **Missing metadata:**
  - radar instrument for Cavitte_2020, Jacobel_2005, Leysinger_Vieli_2011 and Muldoon_2023;
  - acquisition year for Chung_2023 and Franke_2025, and for one Cavitte line ("none");
  - project for the SPRI datasets.
- **Inconsistent names** in the vocabularies (`NASA-CRESIS`, `UTIG-AAD` vs `PRIC-UTIG-AAD`,
  `St-Olaf-College`, "Durham University"). This matters for the filters and for any table in the
  paper.
- **Unit attributes** are only in some datasets' h5 files (section 5.2): recompile or re-set the
  attributes for all.
- **Winter_2018** has two ≈38 ka IRHs (IRH 1 at 38.2 and IRH 5 at 38.1). Check whether this is
  intended (two sub-surveys).
- **BEDMAP1–3** are listed in the docs but not published (section 4.4).
- **The `Distance` totals** include the jumps of lines with gaps, so a surveyed length needs the
  jumps excluded (e.g. steps > some km).

**Code behaviours worth knowing (from a code review in Oct 2026; some are fixed):**
- **wave speed and firn correction:** `CompileDatabase(wave_speed, firn_correction)` also
  multiplies `BED_ELEV` and `ICE_THK` by the wave speed, and adds the correction to them. That is
  correct for a TWT thickness but wrong for an elevation. Check which datasets were compiled with
  TWT (Wang_2023 is the documented example), and whether this affected them.
- Compile-time duplicate positions are dropped (first kept) when the IRHs of a line are merged,
  and the points stay in file order. There is no re-ordering.
- **Query layer, known open issues:**
  - unassigned ages stored as `"nan"` can match numeric filters;
  - INNER JOINs drop files that have no project, institute or radar;
  - lists mixing wildcards and ranges are combined with AND;
  - `filter_out` replaces earlier exclusions.

  See the code-review appendix in the development plans if needed; fix or document these before
  claiming the query semantics in the paper.
- No automated tests or CI tests. A minimal pytest suite (synthetic dataset: compile, index,
  query) would answer a likely reviewer request.

**Scientific scope (state these as limitations or future work):**
- There is no cross-dataset identification of the same isochrone, and no harmonisation of ages
  across datasets or chronologies (in the roadmap: "date layers from the closest/latest
  chronology", "refine isochrone ages for consistent layers across the database").
- Uncertainties: depth uncertainty is not compiled yet (the column is reserved, `IRH_DEPTH_UNC`).
  The age uncertainty is as published, and missing for 4 datasets.
- There is no airborne vs ground-based flag yet (in the roadmap).
- Bedmap3 is used as the fallback surface and thickness, so elevation and fractional depth there
  carry Bedmap3's error and its time mismatch with the survey.
- `BASAL_UNIT` may later become an undated IRH in `IRH_DEPTH` (idea, not done).

**Infrastructure facts to state carefully:**
- the portal's public opening date (section 11);
- the long-term hosting and funding plan, and who maintains it after the PhD;
- a DOI for the software/data (Zenodo releases of the package and of each `db_version`?). None
  exists yet. ESSD, for example, requires the data in a repository with a DOI.

---

## 16. Suggestions for the manuscript

**Journal.** The natural fit is **Earth System Science Data (ESSD, Copernicus)**: a data paper on
a compiled, curated dataset with tools.
- It requires the data to be deposited with a DOI, and a data availability section.
- Cavitte_2020 and Winter_2018 in the database are ESSD papers themselves.
- Alternatives: *Geoscientific Model Development* (if the package and model comparison are the
  focus), *The Cryosphere* (brief communication on the database), *Annals of Glaciology*, or
  *JOSS* for the software alone (short, can complement ESSD).

**Possible structure (ESSD-like):**
1. Introduction: why isochrones matter (ice-sheet history, model constraints, oldest ice); the
   fragmentation of published data; AntArchitecture.
2. Source datasets: table 4.2/4.4 with surveys, radar systems and dating methods. The dating method
   per dataset still has to be collected from the papers: ice-core tie, model, or other.
3. Methods: the standardisation (section 6), the data model (5), ages (7), the index (7).
4. The database: coverage maps, statistics, the age distribution, depth and fractional-depth
   distributions, IRH density.
5. Access: Python package (8), versioned distribution (9), 3D viewer and download page (10).
6. Community contributions: the portal, the checks, the preview and the review (11), with the
   raw archive and provenance.
7. Use cases: model–data comparison with PISM (`book/pism_example.ipynb`, `plot_PISM`); queries
   by age or basin; stratigraphy along a line.
8. Limitations and outlook (15).
9. Data and code availability: bucket and DOI, PyPI, GitHub, docs, website; the licence of each
   source dataset.

**Figures to make, all reproducible from the database:**
- a coverage map coloured by dataset (`db.plot.dataset()`; `book/figures/Overview_all.png`
  exists);
- maps of IRH density (`IRH_NUM`) and of the oldest age per track;
- an age histogram of all IRHs, with uncertainty bars per dataset;
- a 3D viewer screenshot, e.g. Dome C with the isochrones, the ice cores and a stratigraphy
  profile;
- a schematic of the architecture (raw files → compile → h5 + SQLite → bucket/manifest →
  sync/viewer/download; portal → worker → preview → review → publish);
- an example stratigraphy along a line, and a PISM mismatch map.

**Numbers:** compute them with a script, not by hand. The figures in section 4 come from an
ad-hoc script (it iterates the h5 files of the release copy). Make it a `paper_stats.py` in this
repository that writes the tables (CSV/LaTeX) and the figures, so that they can be rerun after
each new dataset or version. Quote the `db_version` used in the paper.

**Licensing to check per dataset:** most sources are PANGAEA, USAP-DC, Zenodo or UK PDC
(typically CC-BY). Confirm each licence allows redistribution, and list them in the paper. The
portal now asks contributors explicitly (section 11.1).
