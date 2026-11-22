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
| EDC | 75°06'02"S, 123°20'52"E, 3233 m | Stauffer et al. (2004), the site selected for drilling | ~150 m | 127 m |
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
- No surveyed position of the borehole itself is published for EDC (the selected site and the camp differ by 60 m),
  Vostok 5G (station level) or DF2 (43 m from DF1). Byrd and Siple Dome are known to about 1 to 2 km only, which limits
  any tie there.

**The radar near the cores** (nearest point with a layer, v2 build, corrected positions): EDC: Bodart_2025b 20 m,
Winter_2018 30 m, Cavitte_2020 110 m, Chung_2023 220 m, Mulvaney_2023 240 m. BELDC: Cavitte_2020, Mulvaney_2023 20 m,
Chung_2023 40 m. Dome Fuji: Wang_2023 10 m, Franke_2025, Winter_2018, Bodart_2025a 120 m. Vostok: Winter_2018 70 m,
Cavitte_2020 80 m, Leysinger_Vieli_2011 600 m. EDML: Franke_2025, Winter_2018 150 m, Bodart_2025a 230 m. TALDICE:
Bodart_2025b 10 m. SPICEcore: Sanderson_2024 630 m, Leysinger_Vieli_2011 750 m, Beem_2021 810 m. WDC: Bodart_2021
520 m, Muldoon_2023 6.1 km. Byrd: Jacobel_2005 310 m, Muldoon_2023 940 m. ABN: Cavitte_2020 3.7 km. None within 20 km:
Taylor Dome, Law Dome, Siple Dome, Skytrain, RICE, Fletcher, Berkner, James Ross Island.

**Where the positions are used:** the regions of the queries (the places of the ice cores: Dome C, Little Dome C, Dome
Fuji, Vostok, Kohnen, Talos Dome, WAIS Divide, Siple Dome; WAIS Divide now has 8 lines instead of 9), the 3D viewer and
the submission form (`web/src/icecore-sites.ts`, checked against the table by `python -m
anta_database.harmonisation.cores`), and the harmonisation.
