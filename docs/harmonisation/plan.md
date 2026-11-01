# Harmonising the layers, and the data structure that follows

Working document, started 2026-11-01. It records the decisions as they are taken, so that the
manuscript (`../manuscript-brief.md`, and the ESSD draft) can be updated. Nothing here is
implemented yet. **Status of each item:**
- *Proposed*: a suggestion, not discussed yet;
- *Agreed*: discussed and accepted;
- *Open*: needs a decision or more data.

Two topics are linked:
1. harmonising the IRHs across datasets: which layers are the same, and one consistent age for
   each;
2. the file format and structure, which must serve the original layers, the harmonised layers,
   and modellers who want one layer at once.

The format study of 2026-11-01 (one netCDF per dataset, CF trajectory ragged arrays, float32) is
in section 6, since the structure depends on the harmonisation.

---

## 1. What the data shows (2026-11-01, published v2 `2026.10.30-1403`)

The figures come from `explore.py` in this directory, which can be rerun on any database copy.

### 1.1 Crossovers between datasets

There are **29,244 crossings** between lines of different datasets, over 33 pairs of datasets.
The lines were thinned to about 100 m and cut at gaps of more than 2 km.

| Pair | Crossings |
|---|---|
| Franke_2025 × Winter_2018 | 16,218 |
| Sanderson_2024 × Winter_2018 | 3,909 |
| Cavitte_2020 × Yan_2025 | 2,029 |
| Bodart_2025b × Winter_2018 | 1,333 |
| Cavitte_2020 × Mulvaney_2023 | 1,289 |
| Bodart_2025a × Franke_2025 | 1,253 |
| Chung_2023 × Mulvaney_2023 | 763 |
| Bodart_2025b × Cavitte_2020 | 710 |
| Bodart_2025a × Winter_2018 | 511 |
| Bodart_2021 × Muldoon_2023 | 313 |
| 23 more pairs | 1 to 200 each |

- **East Antarctica** (Dome C, Little Dome C, Dronning Maud Land, Dome Fuji, Dome A/AGAP, South
  Pole) forms one well-connected network.
- **West Antarctica** (Ashmore, Bodart_2021, Muldoon, Jacobel, Napoleoni, Siegert) is connected
  within itself. It joins the East only through the sparse SPRI lines (Leysinger_Vieli_2011,
  Siegert_2004) and a few crossings. **It will probably be a separate network**, dated by its own
  cores (WAIS Divide, Byrd).

### 1.2 Layers that are probably the same

At the crossovers, the same-looking layers show the following. Depth differences are the median
over the crossings, ± the median absolute deviation.
- **Same tie, same age, small depth difference:**
  - Bodart_2025a vs Franke_2025 (25.6, 38.1, 52.2, 73.4 and 91 ka): **1–3 m**;
  - Cavitte_2020 vs Chung_2023 (73.4, 96.5 and 121 ka): **0–3 m**;
  - Ashmore_2020 vs Bodart_2021 (4.7 ka): 5 m.
- **Systematic depth offsets of 5–20 m between datasets**, larger than the scatter (±1–3 m).
  These pairs always have the same sign, so they are a bias, not noise. The likely cause is
  different wave speeds or firn corrections in the depth conversion.
  - Franke vs Winter: +8 to +9 m at 1,230–1,740 m, for the 38.1 and 73.4 ka layers.
  - Cavitte vs Mulvaney: +6 to +20 m, for layers of nearly the same ages.
  - Bodart_2021 vs Muldoon_2023: −21 ± 12 m at 1,240 m, for the 4.7 ka layer.
- **Near-matches whose ages differ by 1–5 ka:**
  - Mulvaney vs Cavitte: 30.4 vs 29.4, 75.2 vs 73.4, 108.3 vs 106.3 ka;
  - Wang vs Franke: 95.5 vs 91 ka.

  The published ages of the same physical layer differ by chronology (AICC2012 vs others), by
  tie method, and by depth-to-age transfer.
- **Large disagreements to investigate:**
  - Cavitte_2020 vs Yan_2025. Yan uses Cavitte's ages from IRH 10 onwards, yet at 680 crossings
    Yan's 46.4 ka layer is 65 m from the nearest Cavitte layer, and Yan's 29.4 ka layer is 16 m
    shallower than Cavitte's 29.4 ka layer. Possible causes: a different depth convention (firn
    correction or surface reference), a different tracing, or a problem in our compilation.
  - At Dome C below about 1,600 m, Chung vs Mulvaney vs Cavitte have layers of "similar" age
    within 20–30 m of each other, at 300 vs 326 ka, 365 vs 381 ka, and 473 vs 415 ka.

### 1.3 Matching layers by nearest depth fails at depth

Below about 1,500 m at Dome C, consecutive layers are only 20–40 m apart, as close as the
offsets between datasets. The nearest layer in depth is then often the wrong one. For example,
Cavitte IRH 11 (160.4 ka) and IRH 12 (166.4 ka) both come out closest to Chung IRH 8 (160 ka).
**The matching must respect the order of the layers and correct the systematic offsets first**
(section 3).

### 1.4 Ice cores near radar lines with IRHs

| Core | Datasets with an IRH point within 20 km (distance of the nearest point) |
|---|---|
| EDC | Cavitte 0.03 km (19 IRHs), Winter 0.03, Bodart_2025b 0.09, Mulvaney 0.17, Chung 0.19, Leysinger-Vieli 12.9 |
| Little Dome C (BELDC) | Cavitte 0.02 (26 IRHs), Mulvaney 0.02 (20), Chung 0.04 (16), Bodart_2025b 0.35, Winter 0.73, Leysinger-Vieli 1.18 |
| Dome Fuji | Wang 0.01 (7), Franke 0.12 (8), Winter 0.13, Bodart_2025a 0.13 |
| Vostok | Winter 0.07, Cavitte 0.08, Leysinger-Vieli 0.60 |
| EDML | Franke 0.15 (8), Winter 0.15, Bodart_2025a 0.23 |
| TALDICE | Bodart_2025b 0.02 (1 IRH) |
| South Pole (SPICEcore) | Sanderson 0.69, Leysinger-Vieli 0.77, Beem 0.89 |
| WAIS Divide | Bodart_2021 0.49 (3), Muldoon 6.2 |
| Byrd | Jacobel 0.31, Muldoon 0.94 |
| Aurora Basin North | Cavitte 3.6 km |
| Siple Dome, Skytrain, RICE, Law Dome, Taylor Dome, Fletcher, Berkner, James Ross Island | none within 20 km |

Most cores with long records have radar within a few hundred metres, from several datasets. The
core positions are the ones of the viewer (`web/src/icecores.ts`), rounded as published; check
them against the drilling reports before use.

---

## 2. Concepts (*Proposed*)

- **Observation (original IRH):** a layer as traced and published in one dataset, i.e.
  `(dataset, IRH number)`, with its original age and uncertainty. **It is never changed.**
- **Horizon (harmonised layer):** one physical isochrone, with a database-wide id, e.g.
  `AH0123`. One horizon groups the observations of the same layer from several datasets.
  Every observation belongs to exactly one horizon; an observation that matches no other
  dataset is a horizon on its own.
- **Harmonised age** of a horizon: its age with the uncertainty, the method, the chronology and
  the evidence (ice cores, crossovers). It is versioned (section 5).
- **The mapping** observation → horizon carries the evidence:
  - the number of crossovers where they match;
  - the fitted depth offset, and the misfit;
  - a confidence;
  - who confirmed it (automatic or manual).

This separates what was published from what the database adds. Users can always choose the
original age (as in the papers) or the harmonised age (consistent across datasets).

---

## 3. Matching the layers across datasets (*Proposed*)

### 3.1 Inputs to collect first, per dataset

These go in a new metadata table, filled from the papers and kept in the repository:
- **Depth conversion:**
  - the wave speed (m/µs);
  - the firn correction (m), or the density profile used;
  - whether depths are below the surface at the time of the survey;
  - the surface reference (radar surface, or a DEM).
- **Dating:**
  - the method: ice-core tie (which core, which chronology, tie depth at the core, distance of
    the tie), a model (which model, which parameters), or a correlation to another dataset
    (which one);
  - the meaning of the published age uncertainty.
- **Picking:** the radar system and its range resolution, the picking method, and the
  repeat-pick precision, if given.
- **Raw TWT of the layers**, where available (the portal now reserves `IRH_TWT`). With it,
  depths can be reconverted with one common velocity model.

### 3.2 Depth offsets between datasets: a crossover adjustment

As in altimetry and gravimetry, fit one depth correction per dataset,
`z' = z × (1 + s_k) + c_k`: a scale for the wave speed and an offset for the firn correction.
1. Fit it by least squares over all crossovers of the confidently matched layers (same age,
   same core tie, residual < 5 m).
2. Fix one reference dataset, or the mean, to zero.
3. Check the corrections against the wave speeds and firn corrections of 3.1: a scale of
   0.5–1 % is the size of the usual differences in wave speed (0.168–0.1685 m/ns).

The fitted corrections are reported, and used for the matching. *Open:* are the published
depths also corrected in the files (`IRH_DEPTH_HARMONISED`), or only reported? Correcting
reconnects the layers better, but it changes data that users expect to match the publications.
One option: keep `IRH_DEPTH` as published, and give the correction per dataset in the
metadata.

### 3.3 Matching at each crossover, respecting the order of the layers

At a crossover, each dataset gives a sequence of layers sorted by depth. Align the two sequences
with dynamic programming, as in sequence alignment:
- the cost of a match is `(Δz' / σ)²`;
- a layer may stay unmatched, at a fixed cost;
- no crossings are allowed (layer order is kept).

`σ² = σ_pick,A² + σ_pick,B² + (slope × horizontal mismatch)² + σ_offset²`, where:
- `σ_pick` comes from the range resolution, a few metres;
- the slope term is the local layer slope (from the line) times the position error;
- `σ_offset` is the uncertainty of the fitted correction.

The ages enter as a weak prior only: two layers with very different ages need a much better
depth match. This way the matching is not just a copy of the published ages, which are partly
what we want to test.

### 3.4 From crossovers to horizons

- **Link** observations `(A, i)` and `(B, j)` when they are matched at most of their shared
  crossovers (e.g. ≥ 70 %, and ≥ 3 crossovers, or 1 near an ice core). Each link carries its
  statistics.
- **Horizons** are the connected components of the graph of links.
- **Each component is checked:**
  - no two observations of one dataset at different depths (except datasets with duplicate
    layers, e.g. Winter_2018 IRH 1 and 5);
  - stratigraphic order kept against the other horizons;
  - depth residuals around loops of crossovers close to zero.

  Components that fail are split, and flagged for manual review.
- **Manual review** in the 3D viewer: show a horizon in one colour across datasets, with its
  crossovers and their residuals. This is a natural extension of the viewer (a "harmonised"
  colouring mode).

---

## 4. Dating the horizons (*Proposed*)

### 4.1 Principle: interpolate the depth, not the age

At an ice core, the chronology gives the age at each depth exactly; the unknown is **the depth
of the horizon at the core**. So:
1. Estimate the depth of the horizon at the core site from the radar points around it, with an
   uncertainty. Use the depth below the surface, measured from the surface at each point.
2. Read the age at that depth in the chronology: `age = chronology(z_core)`.
3. Uncertainty: `σ_age² = (dA/dz · σ_z)² + σ_chron(z)²`, where `σ_z` combines the
   interpolation, picking and offset uncertainties.

Averaging the *ages* of nearby points is weaker. The age–depth relation is strongly non-linear,
especially at depth, so an average of ages is not the age at the average depth.

### 4.2 Interpolating the depth at the core

- **One line passing over the site** (within ~100 m, the usual case at EDC, LDC, Dome Fuji,
  Vostok and EDML): interpolate along the line at the closest approach.
- **Lines around the site:** fit a local surface of the horizon's depth.
  - It could be a plane or a quadratic within a few km, weighted by distance, or Gaussian process
    regression (kriging), which gives the uncertainty directly. Your earlier tries (distance
    weighting, Gaussian kernels) are special cases of this.
  - The length scale comes from the data: the variogram of the horizon's depth along the lines
    near the core.
- **Fractional depth or depth below the surface?** Near domes the ice thickness can change by
  tens of metres over a few km, so `z / H` can be smoother than `z`. Fit both and compare the
  cross-validation errors: leave one line out and predict it.
- **Surface and bed slopes, and ice flow:**
  - Depths are measured from the local surface, so a sloping surface is already accounted for:
    compare depths below the surface, not elevations. A large difference between the radar
    surface elevation and the core's elevation means a datum or position problem to check, not
    a flow effect.
  - At domes and divides, the ice moves centimetres to metres per year, so a few hundred metres
    of horizontal offset is a small distance in layer geometry. The effect of flow is in the
    shape of the layers (e.g. Raymond arches), which the depth interpolation captures from the
    radar itself.
  - Flow matters only where layers are steep, i.e. away from domes. There the interpolation
    uncertainty, from the local slope and the variogram, is larger, and so is `σ_age`.
- **Core positions:** use the most precise published coordinates. A position rounded to 0.01°
  is up to about 1 km off in longitude near the pole. The sensitivity is the local slope × the
  position error, which goes into `σ_z`.
- **Depth references:**
  - Ice-core depths are below the surface at drilling. Radar depths are below the surface at the
    survey. Years of accumulation between the two shift the depth by a few cm to dm on the
    plateau, and more near the coast. That's negligible in most places; note it.
  - Core depths may also be in "ice equivalent" or "logging depth": use the depth scale of the
    chronology.

### 4.3 Chronologies (*Open*: to collect, versions to verify)

Use the most recent chronology of each core, and keep its name and version with every age. To
check one by one:
- **AICC2023:** EDC, Vostok, EDML, TALDICE (Bouchet et al., 2023).
- **Dome Fuji:** DF2021 (Oyabu et al., 2022), or Dome Fuji on AICC2023 where available.
- **WAIS Divide:** WD2014.
- **South Pole:** SP19 (Winski et al., 2019).
- **Byrd, Siple Dome, Skytrain, RICE, Law Dome, Talos Dome:** their latest published
  chronologies.
- **Beyond EPICA Little Dome C:** probably not published yet. Use EDC through the radar
  connection between EDC and LDC, which is in the data.

Each chronology is stored as a table of depth, age and uncertainty, with its reference, in a
`chronologies/` directory of the database repository, so the dating can be reproduced.

### 4.4 Horizons without a core

1. **Connected to a core through the horizon graph** (section 3): dated by that core. The
   uncertainty adds the depth matching uncertainty along the path of crossovers.
2. **Tied to several cores:** combine the ages, weighted by their uncertainties. Disagreement
   beyond the uncertainties is a result in itself, worth reporting: chronologies or traces
   disagree.
3. **No connection to any core:** keep the original age, flagged `method = original`, possibly
   corrected for the chronology update (4.5). Later, an age from a 1D model, calibrated on the
   dated horizons of the same line (*Open*: which model, e.g. Nye/Dansgaard–Johnsen with fitted
   accumulation and thinning).

### 4.5 Two levels of harmonisation, in that order

1. **Chronology update only:** re-read each published tie in the newest chronology, at the
   published tie depth. This is cheap, easy to explain, and already removes part of the
   differences (e.g. AICC2012 → AICC2023). It needs the tie depths from the papers (3.1).
2. **Full harmonisation:** the matching across datasets (section 3) plus the dating by depth at
   the cores (4.1–4.4).

The manuscript can report both. The table of changes per IRH (section 7) then says how much each
step changed each age.

---

## 5. What the database stores (*Proposed*)

The original data is not touched. New items:

| Item | Content |
|---|---|
| `horizons` | id, harmonised age, uncertainty, method (core tie / via crossovers / original / model), chronology, cores, harmonisation version |
| `irh_horizon` | dataset, IRH, horizon id, original age and uncertainty, number of crossovers, fitted offset, misfit, confidence, reviewed |
| `ties` | horizon, core, chronology, depth at the core ± uncertainty, interpolation method, distance of the nearest point, age ± uncertainty |
| `crossovers` | position, the two lines, the depths of their layers there (for the review and the figures) |
| `depth_corrections` | dataset, scale, offset, uncertainty (section 3.2) |
| `chronologies/` | the age–depth tables used, with references |

- **In the line files,** the `IRH` dimension gets three coordinates: `IRH_AGE_ORIGINAL`,
  `IRH_AGE` (harmonised, or original when there is none) and `HORIZON_ID`, plus
  `IRH_AGE_UNC_ORIGINAL` and `IRH_AGE_UNC`. Because they are coordinates on `IRH`, this costs
  nothing per point.
- **In the SQLite index:** the same tables, so queries can filter on original or harmonised
  ages, horizons, cores, methods or confidence.
- **Versions:** the harmonisation has its own version, e.g. `H1`, written in the manifest and
  in each file. Horizon ids are stable across versions: a re-harmonisation keeps the ids of the
  horizons that still exist.

---

## 6. File format and structure

### 6.1 Format study (2026-11-01)

Measured on all 18 datasets of v2. The scripts are not kept; the results:
- **Today's layout:** one HDF5/netCDF4 file per line, `IRH_DEPTH(point, IRH)`.
  - **629 MB**, i.e. 14.3 bytes per depth value: more than raw float64.
  - Causes: everything in float64, no shuffle filter, gzip level 1, chunks of 100 points × 1
    IRH, and 956 files.
  - The grid is 65 % filled. 479,113 points (5 %) have no IRH, only ice thickness or surface.
  - Points are in along-track order (median step 10–30 m).
  - Reading one IRH of one file from the bucket over HTTP hadn't finished after 4 minutes,
    because of the tiny chunks.
- **One netCDF4 file per dataset**, as a CF-1.11 `featureType = trajectory` contiguous ragged
  array: the lines are concatenated along `obs`, `rowSize(line)` gives their lengths, the
  per-line metadata are `(line)` variables, and `IRH_DEPTH(obs, IRH)`.

| Layout | Size |
|---|---|
| Same encoding as today | 551 MB |
| Float32 values, float64 coordinates, zlib level 4 + shuffle, chunks of 50,000 points | **178 MB** |
| The same with float32 coordinates | 133 MB |
| Zarr v3, float32, zstd | 179 MB |
| Parquet "long" table (one row per depth value, no NaN) | 385 MB |

- **Float32 is effectively lossless here:** depths change by ≤ 0.1 mm and coordinates by
  ≤ 10 cm. The source depths have 0–2 decimals.
- **Tools** (netCDF 4.8.1, NCO, CDO 2.5.4):
  - one netCDF per dataset works with `ncdump`, `ncks` (subset a line with `-d obs,…`) and
    `cdo` (numeric variables only; it skips the strings);
  - **Zarr v2 and v3 fail** with `ncdump`, `ncks` and `cdo`;
  - xarray reads everything.
- **Conclusion:** netCDF4 (= HDF5 underneath), CF ragged arrays, float32, better encoding.
  Not Zarr as the stored format; Zarr references (kerchunk/VirtualiZarr) over the netCDF files
  can be added later for cloud access.

### 6.2 What modellers need, and a structure that serves all queries

The ways of reading the data:
1. **one layer, everywhere:** a harmonised horizon (or an age range), from all datasets, in one
   read;
2. a layer by its original age, or by dataset;
3. **one line with all its layers**, with both ages;
4. a subset by institute, project, region, basin, radar instrument or year.

Ways 1 and 3 are two sort orders of the same samples: by horizon, then position; or by line,
then IRH. No single file layout makes both one contiguous read.

**Proposal: two views of the same data, from one source, sharing one index**

| View | Files | Serves |
|---|---|---|
| **Line view** (the source) | one netCDF per dataset: ragged by line, `IRH_DEPTH(obs, IRH)` and the line metadata | 3, and 2 by dataset; what `sync()` updates dataset by dataset |
| **Layer view** (derived) | one netCDF per horizon: ragged by line. Each point has x, y, depth, elevation, fractional depth, dataset, line, original IRH, original age, and the depth correction applied | 1, and 2 by original age |
| **Index** | SQLite: datasets, lines, horizons, mappings, ties, crossovers, metadata | 4, and finding the right slices for 1–3 |

- **Size:** the layer view holds each depth value once more. That's about +150–180 MB, so
  around **330–360 MB** in all, still well under today's 629 MB.
- **The layer files are regenerated** whenever a dataset or the harmonisation changes.
  Republishing only uploads the horizons that changed.
- **The package API:**
  - `db.horizon(age=38100)` / `db.horizon(id="AH0123")` → one xarray Dataset (all points of the
    layer);
  - `db.line(dataset, line_id)` → one line with all layers, both ages;
  - `db.query(...)` as now, plus `harmonised=True/False` for the ages;
  - `to_netcdf()` / `to_dataframe()` on everything returned.
- **Without Python:** the layer files are plain netCDF, usable from MATLAB, Julia, Fortran, NCO
  or Panoply. A modeller downloads `AH0123.nc` and has the whole layer.

**The alternative considered:** a single "long" columnar table (GeoParquet), partitioned by
dataset and sorted by horizon. It answers every query with predicate pushdown (DuckDB,
pyarrow), even straight from the bucket, with one copy of the data (385 MB measured). But it
isn't readable with the netCDF tools, and modellers mostly use netCDF. *Proposed:* netCDF for the
two views; a Parquet export for analysis can be added later if asked for.

**Gridded products** per horizon (depth, age and uncertainty on a regular grid, e.g. 5 km),
interpolated from the points, would be the most model-ready. They are a separate product, with
choices of method and uncertainty of their own. *Open:* outlook only, or part of this paper?

---

## 7. Documentation and the manuscript

- **docs.antadatabase.com:** a new chapter "Ages and harmonisation", covering the concepts
  (section 2), the method (3–4), the chronologies used, how to choose original or harmonised
  ages in queries, and the versions.
- **The manuscript, sections that change or are new:**
  - data structure and format (section 6: one file per dataset, two views, the new variables);
  - **a new methods section on harmonisation:** depth corrections, matching, dating;
  - **a new results section:** horizons found, datasets connected, ages changed, disagreements
    between datasets or chronologies;
  - access (the layer view, the new API);
  - limitations and outlook.
- **Appendix: one row per original IRH** with the dataset, IRH, original age ± uncertainty,
  horizon, harmonised age ± uncertainty, difference, method, core and chronology, and **where
  the difference comes from** (chronology update, re-tie, crossover correlation, model). Plus a
  table of the depth corrections per dataset. All generated by a script.
- **Figures:**
  - a map of the horizon network (crossovers, cores);
  - original vs harmonised age per IRH;
  - depth residuals at the crossovers before and after the correction;
  - an example horizon across several datasets.

---

## 8. Order of work (*Proposed*)

1. Collect the metadata of 3.1 for the 18 datasets, from the papers: depth conversion and dating
   ties. Fix the DOI and metadata issues listed in the manuscript brief (section 15) at the same
   time.
2. Investigate the large disagreements (Yan vs Cavitte, Bodart_2021 vs Muldoon) and the
   Jacobel_2005 coordinates. Some may be problems in our compilation.
3. Collect the chronologies (4.3) and the precise core positions.
4. Chronology update only (4.5, level 1): the first table of age changes.
5. Crossover adjustment of depths (3.2) and order-preserving matching (3.3–3.4), as a
   prototype in `docs/harmonisation/`. Review in the viewer.
6. Dating by depth at the cores (4.1–4.4), with uncertainties.
7. The new file format (6.1) and the two views (6.2): data schema 3, prefix `v3/`, package 0.7.
   Index, API, viewer and download page adapted.
8. Documentation and the manuscript (7).

Steps 1–6 don't depend on the format, and can start now on the current files. Step 7 waits until
the horizon tables are settled, so that the format is changed only once.

---

## 9. Decisions log

| Date | Decision | Status |
|---|---|---|
| 2026-11-01 | Keep netCDF4 (HDF5 underneath), not Zarr, as the stored format | Proposed |
| 2026-11-01 | One file per dataset, CF trajectory ragged arrays, float32, zlib 4 + shuffle | Proposed |
| 2026-11-01 | Original IRHs and ages never changed; harmonised horizons added alongside | Proposed (from Antoine's request) |
| 2026-11-01 | Date by interpolating the horizon depth at the core, then reading the chronology | Proposed |
| 2026-11-01 | Line view + layer view + one index | Proposed |
| 2026-11-01 | Harmonisation before changing the format (format changed once) | Proposed |
