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
  | NASA_2010_ICEBRIDGE_AIR_BM2 | +4,306 m | surface + thickness |

  **Not feet (checked again 2026-11-10).** In pieces of 1,000 points, the published "surface" is
  a − b × (Bedmap3 surface) with b ≈ 1 and a correlation of about −1: it falls where the real
  surface rises. `a` is about 10,400 m in 2010 (DC-8 flights at high altitude) and 1,500 m in 2004.
  The column holds the distance from the aircraft down to the ice surface, and the "bed" (surface
  + thickness) the distance down to the bed. The aircraft altitude column is empty, so they cannot
  be turned back into elevations. The ratio of 3.35 of 2010 that looked like feet is a coincidence
  of the median: the ratio spans 0.5 to 50. Their surface and bed elevation are left out, and the
  ice thickness kept; the attribute `processing` says so. To report to the Bedmap team.

  Other flags are expected: over ice shelves, the bed is the sea floor and not the base of the ice
  (the 1971 and 1975 seismic surveys of RNRF over the Amery and Filchner-Ronne ice shelves, and
  parts of BAS Dufek 1998, CECS 2006 and NPI MADICE 2016).
- **Parts of surveys found wrong** (`DROPPED_PARTS` in `convert.py`, 2026-11-10), left out with
  all their values, the reason in `processing`:
  - PRIC_2016_CHA2_AIR_BM3, one line of 113 profiles (cut at jumps of 5 km): 3 profiles, starting at
    x, y = (1829, −151), (1056, 228) and (2197, 527) km, 735, 1,143 and 94 km long. Their surface is
    2,672, 1,727 and 729 m below Bedmap3 (median), noisy (3–4 m from one point to the next, 0.3 m
    elsewhere), and their thickness 700 to 1,200 m off. The other 110 profiles are within 30 m of
    the grid. About 30 single points near the South Pole also have thicknesses 300 to 1,700 m off
    the grid; kept for now.
- **Known issues, kept as published** (2026-11-10): BAS_2007_AGAP_AIR_BM2, line 69, after about 445 km: its last
  240 km lie on the Ross Ice Shelf (Bedmap3 surface 50 to 600 m) with the values of the plateau (surface 3,400 m,
  thickness 2,200 to 2,900 m), so its positions look wrong. Kept, to look at in the viewer and report to the Bedmap
  team; no isochrone line is near it.
- **NASA BM2 2002, 2004, 2010 in the viewer:** with their surface left out, the viewer draws their tracks at the
  Bedmap3 surface, as for every point without a measured surface.
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

**Method** (`anta_database/bedmap/reconnect.py`, *Agreed*).
1. Candidates: the Bedmap lines of the same platform whose extent overlaps that of the isochrone
   line (an index of the extents of the 11,168 lines, built once). The platform of a line is the
   `platform` of its raw_md.json (airborne, ground-based, or both); that of a Bedmap survey is in
   its name: `_AIR_` airborne, `_GRN_` and `_SEI_` (seismic) ground-based. Bedmap holds the ground
   surveys of Jacobel_2005 and Napoleoni_2026, but none of the DELORES surveys (Mulvaney_2023,
   Chung_2023): those must not take the airborne lines flown over them. Institute and year are not
   used: many datasets span several institutes and decades, Bedmap dates some surveys by their
   campaign and not their flights (Napoleoni_2026 flown in 2014, in `CECS_2006`), and the traces
   themselves are the evidence.
2. Survey: the one whose lines have the most points within 50 m of the isochrone line. A flight is
   often cut into many Bedmap lines (24 lines of NASA_2018_ICEBRIDGE_AIR_BM3 for one line of
   Bodart_2021), so all the lines of the survey are used. Of surveys with almost as many points
   (90 %), the newest generation: BEDMAP3 over BEDMAP2 over BEDMAP1.
3. Per point: the nearest point of the lines of that survey within 50 m. Values are never mixed
   from two surveys or generations, and never replace a published value.
4. The survey is taken only if its points are within 10 m of those of the line on median
   (`MAX_MEDIAN`, *Agreed* 2026-11-10). The same traces are 0 to 7 m apart; other flights over the
   same place 25 to 42 m (BEDMAP1 against Franke_2025, Winter_2018, Mulvaney_2023). One exception,
   Siegert_2004 (50 m, in `build_v2.py`): BEDMAP1 has its SPRI flights of the 1970s in short pieces,
   34 m away on median, so its fill has gaps.

**Recording it (*Agreed*).**
- `ICE_THK`, `BED_ELEV` and `SURF_ELEV` are filled directly where missing, with the values of the
  Bedmap survey points, never those of the gridded Bedmap products.
- `ICE_THK_SOURCE`, `BED_ELEV_SOURCE`, `SURF_ELEV_SOURCE` (int8, per point, CF `flag_values`,
  2026-11-10): where each value comes from, `0` the published dataset, `1`, `2`, `3` the Bedmap1, 2
  or 3 survey point, `4` computed from the two other published variables (S = B + T); missing
  (`_FillValue` -1) where the variable has no value. One per variable: a line can mix published and
  Bedmap values (Bodart_2021, Cavitte_2020), and a point can take one variable and keep another
  (Leysinger_Vieli_2011 keeps its thickness and takes its bed). The compilation writes 0 and 4, the
  fill 1 to 3. Almost constant, they cost almost nothing compressed.
- `BEDMAP_POINT_DIST` (float32, m, to 0.1 m): the distance to the Bedmap point the values come
  from, NaN where none.
- A `reconnection` attribute names the source, e.g. "BEDMAP3, survey BAS_2010_IMAFI_AIR_BM3, line
  C08a: 99.8% of the points within 50 m (median 1.3 m). Taken from its survey points where the
  dataset has no value, never from the gridded Bedmap products: ICE_THK at ... points", and
  `processing` gets a sentence. Only on the lines filled: a line with nothing missing, or whose
  match is refused, is not changed.
- Run by `docs/harmonisation/build_v2.py` after the compilation, with a table in
  `bedmap_fill.csv`: one row per line with values missing, what was done (`status`: filled, or why
  not) and how many points of each variable were filled.
- A line ID taken from Bedmap would set `line ID flag` to `bedmap`, the earlier name kept in
  `line ID original`. None for now (see the study).

**Study before writing (done, below).** The study matched one Bedmap line per isochrone line; its
coverages are lower than those of the survey rule above where a flight is cut into several lines.

## 7. Viewer (*Proposed*)
- **Bedmap tracks** at the ice surface, one layer per generation, in another colour than the
  isochrone tracks, so radar data not traced yet stands out. The surface is missing on 35–50 % of
  the points: the Bedmap3 grid fills it there, as for the isochrones (`web/irh.py`).
- **Bed lines**: the measured bed along the tracks in the topography colours, or a shadow of the
  tracks on the bed. To try once the tracks exist.
- **Bed confidence**: the bed greyed and darkened as the Bedmap3 `bed_uncertainty` grows, with a
  light hatching where `thickness_survey_count` is 0.

**Study (2026-11-09, `anta_database/bedmap/reconnect.py`, on the lines of v1).** The Bedmap line
with the most points within 50 m of the isochrone line, the newest generation when several have
almost as many (90 %):

| Dataset | Lines matched | Generation | Coverage (median, p10) | Distance (median) | Fills ICE_THK | Own ICE_THK - Bedmap |
|---|---|---|---|---|---|---|
| Ashmore_2020 | 58 / 58, 56 same line ID | BEDMAP3 | 99 %, 95 % | 1.2 m | 784,734 of 807,349 | none |
| Winter_2018 | 76 / 77 | BEDMAP2 (73), BEDMAP1 | 100 %, 17 % | 1.3 m | 662,027 of 816,240 | none |
| Yan_2025 | 31 / 31 | BEDMAP3 | 100 %, 6 % | 0.1 m | 612,565 of 750,741 | none |
| Siegert_2004 | 1 / 1 | BEDMAP1 | 20 % | 34 m | 71 of 348 | none |
| Leysinger_Vieli_2011 | 9 / 9 | BEDMAP1 | 85-100 % on 7 lines | 0.0 m | (has its own) | 4 m median, p95 228 m |
| Cavitte_2020 (control) | 77 / 77 | BEDMAP3 | 100 %, 8 % | 5.6 m | 959 of 12,871 | 0.0 m median, p95 31 m |
| Muldoon_2023 (control) | 107 / 107 | BEDMAP2 | 97 %, 83 % | 3.2 m | (has its own) | -0.6 m median, p95 35 m |

- Where a dataset has its own thickness, Bedmap agrees: median difference 0 to 0.6 m (Cavitte_2020,
  Muldoon_2023). The fill can be trusted where the coverage is high.
- **No line IDs from Bedmap for Siegert_2004 and Leysinger_Vieli_2011**: their traces are in
  BEDMAP1 only (Leysinger_Vieli_2011 at 0.0 m: the same points), which has no line names. They stay
  one line each, `line ID flag` `not_provided`.
- Line names differ between the datasets and Bedmap but for Ashmore_2020 (`c08a` and `C08a`): the
  match is by the traces, the names only confirm it.

**v2 decisions (2026-11-09, *Agreed*).**
- No cutting of lines in v2: `split_into_lines` and `split_at_jump_km` are not used. They stay in
  `compiling/lines.py` and `compile.py` in case they are needed again.
- Siegert_2004 and Leysinger_Vieli_2011, the two datasets without line IDs, are reconnected to
  Bedmap. Bedmap holds their SPRI surveys without line names (BEDMAP3 `STANFORD_1971_SPRI-NSF-TUD`
  is one line, BEDMAP1 a cloud of points): the study says whether another Bedmap line with its own
  name covers them.
- The points of every isochrone line are ordered in compile (section 8). Bedmap lines are not
  reordered: the fill takes the nearest point and does not use the order.

## 8. Open
- Variable `Distance` to be renamed `DIST` in v2, with the isochrone files.
- Reordering points (`compiling/lines.py`) takes 0.1 to 0.2 s per line, also on lines given in
  a scrambled order (BEDMAP3 C08a, 35,000 points; Bodart_2021, Sanderson_2024, Winter_2018,
  Ashmore_2020 lines of up to 41,000 points). It only stalls on a cloud of points with no order
  (BEDMAP1: over 5 minutes on 200,000 points), which is not reordered.
- BEDMAP1 has no line order: kept as one cloud for now.
