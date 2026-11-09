# Dataset issues before harmonising

Checked on 2026-11-01 on the published data, with Antoine's answers. **Fixed** means it is handled in
`build_v1.py` (the v1 build), and nothing is left to do.

## Decided and fixed

- **Names:** datasets keep the year of the **dataset DOI**, not of the paper (Cavitte_2020, Winter_2018,
  Muldoon_2023, Mulvaney_2023, Bodart_2025a and b are correct). Document this rule in the docs.
- **Siegert_2004:** citation corrected to 2004. **Fixed**
- **Bodart_2025a, b:** publication DOI updated to the final paper, `10.5194/tc-20-1379-2026`. **Fixed**
- **Lines without IDs:** named after the dataset, numbered when cut.
  - Leysinger_Vieli_2011: cut at jumps above 50 km into 9 lines (`Leysinger_Vieli_2011_1` to `_9`).
    The steps have a clear gap: largest within a line 30 km, smallest between lines 205 km. **Fixed**
  - Siegert_2004: one straight line, `Siegert_2004`. **Fixed**
- **Winter_2018:** the raw files *do* have real line IDs, in the column `Line` (77 values, e.g.
  `AWI083115`, `099025ast10`), next to `Event`, `Flight` and `Campaign`. Its 27 files became 77 lines.
  `AGAP_South_track` (36,520 km) is now 45 lines. **Fixed**
- **Processing metadata:** attribute `processing` in the files, shown when one dataset is queried. **Fixed**
- **`radar_instrument`** key in Jacobel_2005, Leysinger_Vieli_2011, Muldoon_2023. **Fixed**
- **Cavitte_2020 raw_md.json:** IRH 7 pointed to the file of IRH 6 (a typo). **Fixed**
- **Units** on all variables, including `IRH_AGE`, `IRH_AGE_UNC`. **Fixed**
- **Chung and Mulvaney's close lines; apparent duplicates:** kept as individual lines.

- **Cavitte_2020:** `ICE_THK` and `BED_ELEV` without the +14.6 m, and the published ages. Confirmed by Antoine. **Fixed**
- **Napoleoni_2026:** dataset DOI is the supplement of the article, `10.5194/tc-20-2793-2026-supplement`. **Fixed**
- **Yan_2025:** `_depth` removed from the line IDs and file names. **Fixed**
- **Age uncertainties:** Siegert_2004 from Ashmore et al. (2020), 0.160, 0.175, 0.181, 0.324 ka for the 3.1, 5.6, 6.4,
  16 ka layers; Jacobel_2005 from Bodart et al. (2021), 0.5 ka. None for the 0.8 ka layer, Beem_2021, Leysinger_Vieli_2011. **Fixed**
- **Jacobel_2005, age:** 17.5 ka, as in the raw files of the dataset DOI (column `17.5ka`). Welch and Jacobel (2003) said
  15.0 to 15.5 ka, and the dataset is what we cite.
- **Siegert_2004, DOI:** `10.1002/esp.1238` is the DOI given by the authors. Kept for now.
- **"DOI publication" renamed "DOI article"** in `raw_md.json`, the files, the SQLite index (`DOI_article`), the site and the portal. Older
  `raw_md.json` still read. **Fixed**
- **Missing radar and project:** left empty for now (Cavitte_2020; Leysinger_Vieli_2011, Siegert_2004).
- **BEDMAP1-3:** "coming soon" in `book/intro.md`, until they come back. **Fixed**
- **Vocabularies:** one name for each, `VOCABULARY` in `build_v1.py`, in the files and `raw_md.json`. **Fixed**

  | Field | Before | Now |
  |---|---|---|
  | institute | `UTIG-AAD` | `UTIG`, `AAD` |
  | institute | `PRIC-UTIG-AAD` | `PRIC`, `UTIG`, `AAD` |
  | institute | `NASA-CRESIS` | `CReSIS` |
  | institute | `SPRI-NSF-DTU` | `SPRI-NSF-TUD` (the name in the papers) |
  | project | `ICECAP-OIA` | `ICECAP`, `OIA` |
  | project | `ICECAP2` | `ICECAP` |
  | project | `AGAP-NORTH`, `AGAP-SOUTH-GAMBIT` | `AGAP` |
  | project | `WISE_ISODYN` | `WISE-ISODYN` |
  | radar | `SPRI-NSF-DTU-radar` | `SPRI-NSF-TUD-radar` |

## To check: Antoine

- [ ] **Beem_2021, years:** asked for 1996-2015, but the paper says the SPC survey flew in February 2016 and
      2017, all 15 lines are SPC, and their IDs split the same way (`GCX0d` 2016, `GCX0e` 2017). So the files keep
      2016 or 2017, and `raw_md.json` now says 2016-2017. Project still lists `PPT` (1998-1999), which has no line.
- [ ] **Cavitte_2020, depths below the ice:**
  - line `ASB_JKB1a_R09Ea`, IRH 11 (`raw/OIA_EDC_IRH11.csv`): 15,311 of 110,248 points, typically
    2,016 m deep where the ice is 1,688 m thick (up to 744 m below the bed);
  - line `VCD_JKB1a_X08a`, IRH 18 (163 points, `OIA_EDC_IRH18.csv`) and IRH 19 (386, `OIA_EDC_IRH19.csv`),
    105 and 280 m below the bed.

  The firn correction is not the reason. They look like bad thickness or bad picks.
- [ ] **Napoleoni_2026:** three files have up to 197 jumps (`2014-12-22-P2_COMB_INT_COH4P3`), the
      other 113 none. The citation says "(2026a,b)": which two?
  Found in the raw files (2026-11-09), to report to the authors:
  - `2014-12-22-P2_COMB_INT_COH4P3`: 98 rows are points of other profiles (`2014-12-21-P3_COMB_INT_COH4P2`,
    `2014-12-22-P1_COMB_INT_COH4P4`, `2014-12-17-P1_COMB_INT_COH4`) inserted between its points, each with one
    layer (94 IRH5, 5 IRH6), a depth a few m off, and the surface of the row next to it: all 197 jumps;
  - all 79 positions of `2014-12-21-P3_COMB_INT_COH4P2` are those of points of other files, with other surfaces;
  - single-point spikes: `2014-12-22-P1_COMB_INT_COH4P4` IRH6 rows 15 and 47 (730 m among 517 m, 678 m among
    570 m), `2014-12-23-P2_COMB_INT_COH4P1` IRH5 last row (613 m among 854 m); the end of
    `2014-12-21-P3_COMB_INT_COH4P1` drifts by about 50 m.
- [ ] **Beem_2021:** the paper says 167 m/us, all others 168.5-169.5, and no firn correction. Right?
- [ ] **Winter_2018:** the files of IRH 1 (`CEA_38ka`, 38.2 ka) and 5 (`DML_38ka`, 38.1 ka) are the two regions
      of the paper. Same layer? The harmonisation will test it, like IRH 4 and 10 of Yan_2025.

## Known issues of v1, to fix in v2

- [ ] **Layers listed without values** (found 2026-11-09): 21 of the 1,014 files of v1 list 69 layers that have
      no value on their line: Leysinger_Vieli_2011, all 9 files (44 layers), and Winter_2018, 12 of 77 files
      (25 layers; e.g. `099011ast10`, only in `CEA_161ka.tab`, also lists 38.2, 48.3 and 90.2 ka). No value is
      lost or wrong. Cause: `build_v1.py` (`parts_of`) cuts a file into lines with `isel(point=...)`, which keeps
      every layer of the file. The index takes the ages of a line from `IRH_AGE`, so a query by age returns
      these lines, and `data_generator` yields all-NaN columns (warnings of `np.nanmean`). v1 stays as is; the
      Quick Start skips these lines and says so. In v2: compile from the raw files (it drops empty layers,
      and v2 does not cut lines), `netcdf.write` drops all-NaN layers, and the submission checks flag them.
- [x] **Cavitte_2020, interpolated depths** (fixed for v2, 2026-11-09): the published files fill the gaps of the
      HiCARS picks (the 30 lines `OIA_JKB2n_*`) with interpolated depths. Marie Cavitte's picks of these lines
      (`~/data/isochrones/Cavitte_fixed_data/`, 2025-12-28, IRH 1-20; `HiCARS_Bed.txt` is IRH 1) have the same
      points without the fills, and their depths are the published ones + 10 m (their firn correction) at every
      point in both. `fix_cavitte.py` sets the depths with no pick at the same position (1 m) to nan: 108,942
      depths, on 2 to 19 lines per IRH (most in the deep IRH 15-19). IRH 21-26 and the other
      lines (MCoRDS, DELORES) are not in the picks and stay as published. The corrected files are in
      `~/data/isochrones/AntADatabase-v2-source/Cavitte_2020/raw/`, the source of the v2 compile; v1 keeps them.
- [ ] **Bodart_2021, ice thickness of flight OIB 20181022_01** (found 2026-11-09): in `R2_OIB_20181022_01.csv`
      and `R3_OIB_20181022_01.csv`, `RadarIceThick` repeats the same value at 97-98 % of the steps and is 346 and
      407 m below Bedmap3 (median), while `R1_OIB_20181022_01.csv` and the other 9 files agree with Bedmap3
      (median 0 to 12 m). At a point in several files, v1 took the value of R3 and v2 that of the first file (R1),
      so lines `20181022_01_001_025` and `20181022_01_026_046` change by up to 1,003 m; the points only in R2 or R3
      keep the wrong values in both. Proposed for v2: leave out the thickness of these two files, and fill it
      from Bedmap.
