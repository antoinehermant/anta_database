# Harmonisation: what we do, and what we find

The running record of the harmonisation (plan: `plan.md`, section 8c), written as we go. It is the source of the
methods and results of the manuscript, and of the section "Harmonisation" of the processing page of the website
(`/processing/`). Everything done to the data, every choice and every finding goes here, with its numbers, including
what did not work. The published values of the datasets are never changed; the harmonisation adds to them.

Input: the v2 build of 2026-11-21 (anta_json tag set on 2026-11-23). Code: `anta_database/harmonisation/`.

---

## H1. Ice-core positions

The ages of the layers come from ice cores, and a layer is tied to a core where the radar lines pass near it, so the
position of each core must be known to better than the distance between the radar traces near it (about 10 to 100 m).

**What we found.** The positions used so far came from two tables of the package that disagreed: the table of the
regions (`ice-core-coordinates.csv`) and the one of the 3D viewer. The first had errors of 20 to 38 km (Vostok, WAIS
Divide, Talos Dome); the second was closer, but rounded to 0.01° to 0.0001° for some cores (0.01° of latitude is about
1.1 km).

**What we did.** One table of the positions, from the primary sources (drilling and logging reports, the data records
of the cores), with its source and precision for each core, used everywhere in the database: the regions of the
queries, the 3D viewer, the submission form and the harmonisation.

**The table** (`anta_database/data/ice_cores.csv`, H1 done on 2026-11-22). "Moved": the distance from the position
used by the viewer before; precision: what the source allows, not a measured error.

| Core | Position | Source | Precision | Moved |
|---|---|---|---|---|
| EDC | 75.10162°S, 123.34820°E, 3233 m | The sign at the drill tent (photograph, personal communication) | ~10 m | 9 m |
| BELDC | 75°17'57.02"S, 122°26'42.5"E, 3230 m (WGS84) | Lilien et al. (2021) | ~10 m | 3 m |
| Dome Fuji | 77°19'01"S, 39°42'12"E, 3810 m | Dome-F Deep Coring Group (1998), DF1; DF2 is 43 m north | ~50 m | 85 m |
| Vostok | -78.46442, 106.83732, 3488 m | PANGAEA (AICC2012 Vostok) | ~300 m | 3 m |
| EDML | -75.0025, 0.0684, 2891.7 m (WGS84) | Wesche et al. (2007), GPS survey | ~10 m | 0 m |
| TALDICE | 72°49'40"S, 159°11'00"E, 2315 m | Frezzotti et al. (2004), Urbini et al. (2006), site ID1 | ~30 m | **1,230 m** |
| Taylor Dome | 77°47'47"S, 158°43'26"E, 2365 m | UW IsoLab | ~30 m | 1 m |
| Law Dome (DSS) | 66°46'11"S, 112°48'25"E | Jong et al. (2022) | ~30 m | **369 m** |
| SPICEcore | 89.9889°S, 98.1596°W, 2835 m | Casey et al. (2014) | ~50 m | 120 m |
| WDC | 79°28.0580'S, 112°05.1890'W, 1766 m (WGS84) | Fitzpatrick et al. (2014) | ~10 m | 75 m |
| Byrd | 80°01'S, 119°31'W, 1530 m | Ueda and Garfield (1969) | ~1 km | 4 m |
| Siple Dome | 81.65°S, 148.81°W, 621 m | Taylor et al. (2004) | ~2 km | 0 m |
| Skytrain | 79°44.46'S, 78°32.69'W, 784 m | Hoffmann et al. (2022) | ~20 m | 1 m |
| ABN | -71.166889, 111.366531, 2690 m | AADC, main hole of ABN1314 | ~10 m | **369 m** |
| RICE | -79.364, -161.706, 550 m | PANGAEA (Bertler et al., 2017) | ~110 m | **446 m** |
| Fletcher | 77°54.10'S, 82°36.30'W, 873 m | Mulvaney et al. (2014) | ~200 m | 216 m |
| Berkner | 79°32.90'S, 45°40.70'W, 890 m | Mulvaney et al. (2007) | ~200 m | 186 m |
| JRI | 64°12.10'S, 57°41.10'W, 1542 m | Mulvaney et al. (2014) | ~200 m | 4 m |

The old table of the regions was much further off: Talos Dome 37.6 km, Vostok 21.7 km, WAIS Divide 20.6 km.

**Pitfalls found in the sources**, worth a sentence in the manuscript:
- Some data records give another core's position under the same site name: the PANGAEA AICC2012 TALDICE event is the
  1996 firn core near the summit (-72.78333, 159.06667), not TALDICE; PANGAEA's Dome Fuji longitude is 39°40' instead of
  39°42' (0.8 km).
- Rounded values propagate: the 72°49'S of the TALDICE website (for 72°49'40"S) is 1.2 km north; the 0.01° of drilling
  reports and data centres is up to 1.1 km.
- No surveyed position of the borehole itself is published for EDC: the site selected for drilling (Stauffer et al.,
  2004) is 118 m from the borehole, and the camp (Vittuari et al., 2004) 60 m from that site. We use the position on the
  sign at the drill tent, given by a colleague from a photograph; it puts the Cavitte_2020 line 32 m from the core
  instead of 109 m. Nor is one published for Vostok 5G (station level) or DF2 (43 m from DF1). Byrd and Siple Dome are known to about 1 to 2 km only, which limits
  any tie there.

**The radar near the cores** (nearest point with a layer, v2 build, corrected positions): EDC: Winter_2018 30 m,
Cavitte_2020 30 m, Bodart_2025b 80 m, Mulvaney_2023 170 m, Chung_2023 190 m. BELDC: Cavitte_2020, Mulvaney_2023 20 m,
Chung_2023 40 m. Dome Fuji: Wang_2023 10 m, Franke_2025, Winter_2018, Bodart_2025a 120 m. Vostok: Winter_2018 70 m,
Cavitte_2020 80 m, Leysinger_Vieli_2011 600 m. EDML: Franke_2025, Winter_2018 150 m, Bodart_2025a 230 m. TALDICE:
Bodart_2025b 10 m. SPICEcore: Sanderson_2024 630 m, Leysinger_Vieli_2011 750 m, Beem_2021 810 m. WDC: Bodart_2021
520 m, Muldoon_2023 6.1 km. Byrd: Jacobel_2005 310 m, Muldoon_2023 940 m. ABN: Cavitte_2020 3.7 km. None within 20 km:
Taylor Dome, Law Dome, Siple Dome, Skytrain, RICE, Fletcher, Berkner, James Ross Island.

**Where the positions are used:** the regions of the queries (the places of the ice cores: Dome C, Little Dome C, Dome
Fuji, Vostok, Kohnen, Talos Dome, WAIS Divide, Siple Dome; WAIS Divide now has 8 lines instead of 9: the old table put the core 20.6 km north-east, at 79.28°S, 112.05°W, so two
Muldoon_2023 lines 33 and 46 km from that point, but 54 and 66 km from the core, left the 50 km circle, and one 41 km
from the core came in), the 3D viewer and
the submission form (`web/src/icecore-sites.ts`, checked against the table by `python -m
anta_database.harmonisation.cores`), and the harmonisation.

---

## H2. Chronologies

Every age of the harmonisation comes from a published depth-age table of an ice core. They come in many formats
(PANGAEA, NOAA, data papers), with ages before 1950 or 2000, in years or thousands of years, and with
uncertainties at 1 or 2 sigma or none at all.

**What we did.** One reader per published file (`anta_database/harmonisation/chronologies.py`). Each one gives the
same table: depth (m, on the depth scale of the core's log), ice age (ka before 1950), and uncertainty (ka, 1 sigma, NaN
where none is published). Every table is checked: the age must increase with depth. The source files are read where
they were downloaded, with their origin, and are not redistributed until their licences are checked (SP19 is
CC BY-NC). The tables are written to `~/data/isochrones/harmonisation/chronologies/<core>/<name>.csv`, with
`registry.json` (H2 done on 2026-11-22).

| Chronology | Core | Depth (m) | Age (ka) | Uncertainty | Source |
|---|---|---|---|---|---|
| **AICC2023** | EDC | 0-3193 | 0-813 | 1 σ | Bouchet et al. (2023), PANGAEA.961017 |
| **AICC2023** | EDML | 18-2416 | 0-145 | 1 σ | idem |
| **AICC2023** | TALDICE | 0-1597 | 0-492 | 1 σ | idem |
| **AICC2023** | Vostok | 0-3272 | 0-408 | 1 σ | idem |
| AICC2012 | EDC, EDML, TALDICE (to 207 ka), Vostok | | | none in these files | the AICC2012 column of the AICC2023 files |
| EDC3 | EDC | 0-3189 | 0-802 | none | Parrenin et al. (2007), NOAA |
| WD2014 | WDC | 0-3404 | 0-67.75 | 2 σ, halved | Buizert et al. (2015), Sigl et al. (2016), NOAA |
| SP19 | SPICEcore | 0-1751 | 0-54.3 | total, halved | Winski et al. (2019), USAP-DC 601206 |
| DF2021 | Dome Fuji | 0-2150 | 0-208 | 2 σ, halved | Oyabu et al. (2022), NIPR ADS |
| DFO-2006 | Dome Fuji | 0-2504 | 0-342 | none | Kawamura et al. (2007), NOAA (b2k, converted) |
| GT4 | Vostok | 0-3310 | 0-423 | none | Petit et al. (1999), NOAA |
| Byrd-BB2001 | Byrd | 940-2142 | 10-90 | none | Blunier and Brook (2001), NOAA (GISP2 scale) |
| Vostok-RR2003 | Vostok | 150-3165 | 6-349 | none | Ruddiman and Raymo (2003), NOAA: **our reconstruction** of the ice age (below) |
| AICC2023 | NGRIP | 8-3082 | 0-125 | 1 σ | Bouchet et al. (2023) |
| GICC05 | NGRIP | 7-2427 | 0-60 | maximum counting error, as 2 σ, halved | Svensson et al. (2008), NOAA |
| GICC05 | GRIP | 5-2097 | 0-32 | idem | idem |

The Greenland chronologies are there for reference: for the uncertainty of AICC2023 (below), and for Greenland
isochrones later. The Greenland cores are not in `ice_cores.csv` yet.

From WD2014 to AICC2023 we use the 395 volcanic ties between WDC and EDC of Bouchet et al. (2023), valid from 10 to
57 ka (`wd2014_to_aicc2023`). The difference AICC2023 − WD2014 is +65 years at 20 ka, −136 at 40 ka and −312 at 57 ka.

**How much the chronologies of a core differ** (age of the same depth, ka), i.e. what a change of scale alone does to
the age of a layer:

| | 500 m | 1000 m | 1500 m | 2000 m | 2500 m | 3000 m |
|---|---|---|---|---|---|---|
| EDC: AICC2012 − AICC2023 | +0.24 | +0.43 | −2.27 | +0.85 | +0.05 | −1.24 |
| EDC: EDC3 − AICC2023 | −0.33 | −0.67 | −1.49 | −1.64 | +0.15 | −1.50 |
| Vostok: GT4 − AICC2023 | −3.17 | +0.13 | −2.39 | −1.01 | +1.11 | −13.0 |
| Dome Fuji: DFO-2006 − DF2021 (300, 600, 1000, 1500, 2000 m) | +0.19 | −0.23 | −1.96 | +0.21 | −0.79 | |

**What we found, and what is still open:**
- **Over the last 60 ka, the uncertainty of AICC2023 is relative to the Greenland layer count.** Bouchet et al. (2023,
  section 2.1) hold NGRIP exactly to the layer-counted GICC05 by absolute tie points every metre over the last 60 kyr.
  So AICC2023 equals GICC05 at NGRIP within a year (checked: 35.087 ka at 2000 m in both), and its uncertainty there
  (17 years at 35 ka at NGRIP, 20 to 100 years at EDC between 30 and 50 ka) leaves out the counting error of GICC05
  (650 years at 35 ka and 1.25 ka at 58 ka, 1 σ). Bouchet et al. also recommend timescales tied to WD2014 over the last
  60 kyr. *Proposed:* for a layer younger than 60 ka, add the counting error of GICC05 at its age to the AICC2023
  uncertainty (in quadrature), and give its WD2014 age too (through the volcanic ties).
- **Leysinger Vieli et al. (2011) at Vostok.** The paper (paragraph 17) dates the Vostok layers with "the ages
  calculated by Ruddiman and Raymo [2003]". It uses Parrenin et al. (2007), i.e. EDC3, at EPICA, and there the layers
  are dated through the normalised depth on a line 13 km from the core. But Ruddiman and Raymo (2003) is a gas-age
  scale: their Table 1 has 33 tie points of depth and gas age, and no ice-age table is published. We reconstruct the
  ice age the way it was most likely done: their gas age at a depth (linear between the ties) plus the ice minus gas
  age of GT4 there (the delta age they started from). The steps of the GT4 delta age make the sum go back by up to
  50 years at 23 depths; we keep the running maximum. Between 1000 and 2000 m it is within 1.2 ka of GT4; it differs
  by −6.3 ka at 2500 m and +12.8 ka at 3000 m. If it doesn't give their published ages back in H3, we document it and
  leave the dating of that dataset to the AICC2023 ties of the other datasets.
- **AICC2012 uncertainties** are not in the AICC2023 files (Bazin et al., 2013 published them separately). They're only
  needed to compare with the papers that used AICC2012.
- **Depth scales.** DF2021 is on the DF1 depth (Oyabu et al., 2022: the age markers are given at "ice depth (m, DF1)",
  and those measured on DF2 or DF2001 are converted to DF1 depth). So are DFO-2006 and our position of Dome Fuji. At
  Vostok the record is a composite of several boreholes (3G, 4G, 5G, some hundreds of metres apart, matched by their
  depths); AICC2012 and AICC2023 don't say on which depth log, and the cores are within the precision of our position
  (300 m). The difference between the logs is a few metres at most, small next to the depth of a layer from the
  radar.
- **Not used by any dataset, not read:** EDML1, TALDICE-1, ST22, RICE17, Byrd-H1994, DFO-2006+AICC2012 (they're in
  `chronologies.json`). AICC2012 below 342 ka at Dome Fuji (Uemura et al., 2018) isn't read either; no layer is that
  deep there.
