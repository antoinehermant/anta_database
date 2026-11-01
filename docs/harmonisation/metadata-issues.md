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
- [ ] **Beem_2021:** the paper says 167 m/us, all others 168.5-169.5, and no firn correction. Right?
- [ ] **Winter_2018:** the files of IRH 1 (`CEA_38ka`, 38.2 ka) and 5 (`DML_38ka`, 38.1 ka) are the two regions
      of the paper. Same layer? The harmonisation will test it, like IRH 4 and 10 of Yan_2025.
