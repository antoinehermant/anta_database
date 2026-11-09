# Bedmap survey lines in the database

Working document, started 2026-11-09, written to become a section of the manuscript
(`../manuscript-brief.md`). Status of each item: *Agreed*, *Proposed* or *Open*, as in
`../harmonisation/plan.md`.

## 1. Why
BEDMAP1, BEDMAP2 and BEDMAP3 (Frémand et al. 2023, https://doi.org/10.5194/essd-15-2695-2023)
publish the radar survey lines behind the Bedmap grids: surface elevation, ice thickness and bed
elevation along each line, with no IRHs. They serve the database three ways:
1. **filling the isochrone lines**: ice thickness and bed elevation where a dataset does not
   publish them (Ashmore_2020, Siegert_2004, Winter_2018 and Yan_2025 have neither), and a line ID
   for datasets without one (Siegert_2004, Leysinger_Vieli_2011);
2. **the extent of radar data not traced yet**, next to the isochrone tracks in the viewer;
3. **where the Bedmap3 bed is measured and where it is interpolated**, from the
   `thickness_survey_count` and `bed_uncertainty` of the Bedmap3 grid.

They are part of the **v2** database (*Agreed*): v1 stays untouched. The viewer can show them
before v2 is public, since its data has its own prefix in the bucket (`web/`).

## 2. Source files
The standardised CSV files of Bedmap, one per survey (CC-BY 4.0):

| | Surveys | Points | CSV |
|---|---|---|---|
| BEDMAP1 | 1 (all surveys 1966–2000) | 1.9 M | 0.16 GB |
| BEDMAP2 | 66 | | 2.7 GB |
| BEDMAP3 | 84 | | 6.8 GB |

Each file starts with `#name: value` lines (project, institution, instrument, wave speed in ice,
firn correction, licence...), then the columns `trajectory_id`, `trace_number`, longitude,
latitude, date, time, surface altitude, ice thickness, bed altitude, two-way travel time, aircraft
altitude and along-track distance, with `-9999` for missing values.

A survey can be in two generations: the IMAFI survey of BAS (2010) is in BEDMAP2 (lines `7`, `10`,
`11`...) and in BEDMAP3 (lines `C07d`, `C08a`...). **Both generations are kept** (*Agreed*);
reconnection takes the most recent one (section 6).

## 3. The earlier compilation, and what it got wrong
Bedmap was first compiled with the isochrone pipeline into one h5 file per line, float64, named by
line ID. A check of these files against the CSV files (2026-11-09) found:
- **Line names lost.** When `raw_md.json` gave a survey a single "flight ID", the whole survey
  became one line under that name. This was needed where `trajectory_id` numbers the points (one ID
  per point: BEDMAP1, and 55 of the 66 BEDMAP2 surveys), but 17 BEDMAP3 surveys were treated the
  same way, and some of them had real line names: KOPRI KRT1 (2017) has 63 named lines
  (`CBG/CBG03b`, `DVG/X25a`...).
- **Two lines merged.** Files were named by line ID alone, in one directory per generation. Line
  `C9` exists in the Lake Ellsworth surveys of 2007 (1,525 points) and 2008 (4,767 points); both
  were written to one file (6,258 points), labelled 2008.
- Otherwise **no points were lost**: lines found in both have 99.5 to 100 % of the points of the CSV.
- Float line IDs: the South Pole Lake survey (2018) names its lines `1.0`, `2.0`...

## 4. Conversion: `anta_database/bedmap/convert.py`
The lines are now converted from the CSV files directly (*Agreed*):
- **Lines.** A survey's `trajectory_id` is taken as line names when each ID has at least 10 points
  (median). Every survey checked has its lines as single contiguous runs. Names are kept as
  published, `/` included (`CBG/CBG03b`); `3.0` becomes `3`. `line ID flag` is `original`.
- **Surveys without line IDs** (all `-9999`, e.g. AWI GEA-IV 2013, PRIC CHA1–4, UTIG GIMBLE), or
  with point numbers, are one line named after the survey (`GEA-IV_AIR_2013`), with `line ID flag`
  `not_provided`. They are not cut at jumps for now (*Agreed*). Cutting at jumps is an option
  (`--split-at-jump-km`): at 10 km, PRIC CHA2 gives 114 lines, of which 28 are single points.
- **Order.** Points stay in the order of the file, which is the order of the traces. A line is
  flagged when more than 5 of its steps are over 50 times the usual spacing and over 2 km. In a
  sample of 300 lines of the earlier files, 1 BEDMAP3 line and 6 BEDMAP2 lines are flagged.
  BEDMAP1 is a cloud of points in no line order (about 32,000 such jumps).
- **File names** are `<survey>__<line>.nc` (`BAS_2007_Lake-Ellsworth_GRN_BM3__C9.nc`), so lines of
  different surveys never meet.
- **Values as published.** Surface elevation, ice thickness and bed elevation are stored when
  measured, and never computed from the two others (*Agreed*, section 5). Points with none of the
  three are dropped.
- **Format.** That of the isochrone lines (`anta_database/netcdf.py`): netCDF-4, values in float32,
  coordinates and Distance in float64, zlib 4 with shuffle. The attributes are those of the
  isochrone lines (`dataset`, `institute`, `project`, `radar instrument`, `acquisition year`,
  `line ID`, `line ID flag`, `DOI dataset`, `DOI article`, `processing`, `IMBIE_basins`), plus
  `survey`, the name of the CSV file.
- **Result (2026-11-09)**, in `~/data/isochrones/BEDMAP-build/`:

  | | Surveys | with line IDs | Lines | Points | Size |
  |---|---|---|---|---|---|
  | BEDMAP1 | 1 | 0 | 1 | 1.9 M | 33 MB |
  | BEDMAP2 | 66 | 12 | 2,595 | 31.0 M | 630 MB |
  | BEDMAP3 | 84 | 71 | 8,572 | 61.5 M | 1.4 GB |

  216 lines are flagged as maybe out of order: 152 in BEDMAP3, of which 57 are in LDEO ROSETTA
  (2015) and 22 in AWI OIR (2016), and 64 in BEDMAP2, mostly whole surveys stored as one line.
- **Values found wrong in the published files** (`DROPPED` in `convert.py`). Each survey was
  compared with the Bedmap3 grid (surface, and bed = surface − thickness). Three surveys of
  BEDMAP2 fail, while their ice thickness agrees with the grid (median difference 0 to 4 m):

  | Survey | Surface − Bedmap3 (median) | Bed in the file |
  |---|---|---|
  | NASA_2002_ICEBRIDGE_AIR_BM2 | +440 m | surface + thickness |
  | NASA_2004_ICEBRIDGE_AIR_BM2 | −403 m | surface + thickness |
  | NASA_2010_ICEBRIDGE_AIR_BM2 | +4,306 m (+67 m once read as feet) | surface + thickness |

  Their surface and bed elevation are left out, and the ice thickness kept; the attribute
  `processing` says so. To report to the Bedmap team. Other flags are expected: over ice shelves,
  the bed is the sea floor and not the base of the ice (the 1971 and 1975 seismic surveys of RNRF
  over the Amery and Filchner-Ronne ice shelves, and parts of BAS Dufek 1998, CECS 2006 and
  NPI MADICE 2016).
- **Metadata.** `raw_md.json` has one row per survey with the current keys and
  `"file structure": "survey"`. The radar instrument, wave speed and firn correction come from
  the header of the CSV: the instrument is known for every survey of BEDMAP2 and BEDMAP3, where the
  earlier files had none.

## 5. Three variables, not two (*Agreed*)
Surface elevation = bed elevation + ice thickness on grounded ice, so one of the three could be
left out. All three are kept, as published:
- **Measured values would be lost.** On 150 sampled lines per generation (BEDMAP1: the whole
  file), 35 % (BEDMAP3), 47 % (BEDMAP2) and 50 % (BEDMAP1) of the points have only one of the three,
  e.g. a surface without a bed pick.
- **Little to save.** On 40 BEDMAP3 lines, dropping the bed elevation saves 9 % of the compressed
  size, since missing values compress to almost nothing. Float32 and shuffle save about 50 %.
- **The relation is not always exact.** Where all three exist, |S − T − B| is below 0.1 m for 99 %
  of the points of BEDMAP2 and BEDMAP3, but over 1 m for 4 % of the points of BEDMAP1.

The same rule applies to the isochrone datasets. A helper can compute the third variable on reading,
marked as computed. *Open*: compile still fills `ICE_THK` with `SURF_ELEV − BED_ELEV` when a
dataset has no thickness (`compile.py`, `extract`).

## 6. Reconnecting the isochrone lines to Bedmap (*Proposed*, for v2)
**Test (2026-11-09).** Line `c08a` of Ashmore_2020 (BAS IMAFI 2010, no thickness or bed) against
line `C08a` of BEDMAP3 IMAFI:
- nearest Bedmap point: median 1.3 m, 95 % within 3.9 m, at most 107 m;
- 99.8 % of the points get an ice thickness within 50 m;
- same spacing (about 9.7 m): the same traces;
- the deepest IRH lies at 49 % of the ice thickness.

**Method.**
1. Candidate surveys: same institute (with aliases, e.g. NASA-CRESIS and CRESIS), acquisition year
   within one year, and overlapping bounding boxes; otherwise every overlapping survey.
2. Line: the same line ID (case-insensitive), else the closest line.
3. Per point: the nearest point of that one Bedmap line within 50 m (*Agreed*). Values are never
   mixed from two lines or generations; BEDMAP3 is preferred over BEDMAP2 over BEDMAP1.

**Recording it (*Proposed*).**
- `ICE_THK`, `BED_ELEV` and `SURF_ELEV` are filled directly where missing (*Agreed*). For each,
  a flag variable `<VAR>_SOURCE` (int8, CF `flag_values` 0 1 2 3, `flag_meanings` "dataset bedmap1
  bedmap2 bedmap3") says where each value comes from.
- `BEDMAP_DIST` (float32, m) is the distance to the Bedmap point used, so values not taken at the
  same point are documented.
- A `reconnection` attribute names the source, e.g. "BEDMAP3, BAS_2010_IMAFI_AIR_BM3, line C08a,
  99.8 % of points within 50 m, median 1.3 m", and `processing` gets a sentence.
- A line ID taken from Bedmap sets `line ID flag` to `bedmap` (*Agreed*), and the earlier name is
  kept in `line ID original`. The documentation states that these line IDs are not those of the
  published isochrone dataset.

**Study before writing (*Open*).** Ashmore_2020, Winter_2018, Yan_2025 and Siegert_2004 (no
thickness), and a control dataset with its own thickness (Cavitte_2020 or Muldoon_2023) to compare
the fill with it; line IDs for Siegert_2004 and Leysinger_Vieli_2011.

## 7. Viewer (*Proposed*)
- **Bedmap tracks** at the ice surface, one layer per generation, in another colour than the
  isochrone tracks, so radar data not traced yet stands out. The surface is missing on 35–50 % of
  the points: the Bedmap3 grid fills it there, as for the isochrones (`web/irh.py`).
- **Bed lines**: the measured bed along the tracks in the topography colours, or a shadow of the
  tracks on the bed. To try once the tracks exist.
- **Bed confidence**: the bed greyed and darkened as the Bedmap3 `bed_uncertainty` grows, with a
  light hatching where `thickness_survey_count` is 0.

## 8. Open
- Variable `Distance` to be renamed `DIST` in v2, with the isochrone files.
- Reordering points (`compiling/lines.py`) takes 0.1 to 0.2 s per line, also on lines given in
  a scrambled order (BEDMAP3 C08a, 35,000 points; Bodart_2021, Sanderson_2024, Winter_2018,
  Ashmore_2020 lines of up to 41,000 points). It only stalls on a cloud of points with no order
  (BEDMAP1: over 5 minutes on 200,000 points), which is not reordered.
- BEDMAP1 has no line order: kept as one cloud for now.
