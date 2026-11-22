# Harmonising the layers, and the data structure that follows

Working document, started 2026-11-01. It records the decisions as they are taken, so that the
manuscript (`../manuscript-brief.md`, and the ESSD draft) can be updated. **The steps to follow are in
section 8c (the harmonisation, from 2026-11-22), the fixes left before it in 8a, the remaining challenges and decisions in 8b.** Status of each item:
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

There are 29,244 crossings between lines of different datasets, over 33 pairs of datasets. The
lines were thinned to about 100 m and cut at gaps of more than 2 km.

**This count is inflated, and the per-pair numbers below are not independent crossovers.** They
fall in only **10,193 distinct 1 km cells**. Two causes:
- **Files holding several flights** (`metadata-issues.md`, section B): one file of Winter_2018 is a
  whole survey of 36,520 km. Cavitte_2020 × Yan_2025 is 2,029 crossings from a *single* pair of files,
  and Sanderson_2024 × Winter_2018 has 3,786 of its 3,909 from one pair.
- **The same flights traced twice:** 57 % of the Chung_2023 points lie within 50 m of a Mulvaney_2023
  point, 18 % of Bodart_2025a in Franke_2025, 17 % of Mulvaney_2023 in Cavitte_2020, 11 % of Chung_2023
  in Cavitte_2020, 9 % of Sanderson_2024 in Winter_2018, 6 % of Bodart_2021 in Muldoon_2023. Some file
  names are shared too (Cavitte/Mulvaney: 6, Bodart_2025a/Franke: 2).

  **Close tracks are not the same data** (A, 2026-11-11): Chung_2023 and Mulvaney_2023 lie very close to each
  other but are different datasets, and must never be treated as the same lines or merged. Their crossings
  and close tracks are counted as crossovers like any other pair, grouped per cluster so that they do not
  outweigh the rest.

**Coincident tracks are the best comparison we have:** the same radar positions, thousands of
points long, rather than one point per crossing. The matching must use both, and count crossovers per
cluster, so one pair of lines cannot outweigh the rest.

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
  - **Cavitte_2020 vs Yan_2025: resolved, mostly a false alarm.** On the flight both have
    (`20131127_01`, all the points within 50 m), Yan's 29.4 ka layer is 15.9 ± 2.3 m shallower. Cavitte has the
    +14.6 m of the compilation, and Yan describes no firn correction, so they differ by only 1.3 m. Yan's "46.4 ka
    65 m away" came from my comparison: Cavitte has no 46 ka layer on that flight, so it picked the next layer.
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

### 3.2 Correcting the depths (*Proposed*, see the answer to "can we do it?")

A radar depth is `z = v · (t − t_surf) / 2 + f`, with `v` the wave speed in ice and `f` the firn
correction. **We can correct exactly what we know, and only that.**

**Level 1, physical: reconvert to one common model.** With the `v` and `f` a dataset used,
`z' = (z − f) · v' / v + f'`, or directly from the travel time where it is published.
- Needs, per dataset: `v`; whether `f` is constant, a profile, or varies in space (and its value);
  the surface used; the travel times, where published. **Collecting them is step 1.1.**
- Needs, once: the common `v'`, and the common firn model `f'(x, y)`. A constant `f'` is wrong across
  a continent, so it should be a firn-air-content map from a firn model (*Open*: which one; the choice
  is a decision for the paper). Ice-core density profiles can check it at the cores.
- Written to the files as `IRH_DEPTH_H`, next to the untouched `IRH_DEPTH`. Reversible, and every
  parameter is documented.
- **It works only for the datasets whose parameters are known.** Those with a variable firn
  correction that is not published can't be reconverted exactly; they get a flag, and level 2 only.

**Level 2, empirical: the residual offsets at crossovers.** Fit one `z' = z × (1 + s_k) + c_k` per
dataset by least squares, as in a crossover adjustment in altimetry, over the confidently matched
layers (same age, same core tie, a small misfit), counted per cluster (section 1.1).
- What is left after level 1 isn't all conversion: picking conventions (peak or leading edge), the
  frequency of the radar, positioning, and the years between surveys (accumulation of a few m on
  the plateau between the 1970s and the 2010s, more near the coast).
- **Used for the matching, and reported per dataset; not written into the depths.** Forcing datasets
  together empirically would hide real differences, and needs an arbitrary reference dataset.

**The test of level 1 (done 2026-11-01, `check_depth_correction.py`).** With the values of
`datasets_methods.csv`, on 71 pairs of layers of about the same age at crossovers, the offsets predicted from
the wave speeds and firn corrections do **not** explain the measured ones:

| | |
|---|---|
| Median of the measured offsets | 8.8 m |
| Median after removing the predicted offsets | 12.3 m |
| Correlation of measured and predicted | −0.18 |

- It works in some cases: Cavitte vs Yan at 29.4 ka (−15.9 m measured, −14.6 m predicted), and Sanderson, Chung and
  Wang vs Winter on some layers.
- It fails in the largest systematic ones. Franke vs Winter: +8.9 m measured, −3.6 m predicted. Cavitte vs
  Mulvaney: +8 to +19 m measured, −4.6 m predicted.
- **Caution:** the pairs are matched by nearest depth, so some are different layers, above all at depth. This must be
  repeated on the horizons of the order-preserving matching (3.3).

**So, for now:** `IRH_DEPTH` stays as published, and `IRH_DEPTH_H` (*Agreed* naming) is only written once the
test passes on matched horizons. The conversion parameters alone will not reconcile the datasets. Other causes
to examine: the picks (peak or leading edge), the radar systems, and the surface pick.

### 3.2b Common parameters (*Proposed*)

- **Wave speed: 168.5 m/us.** 10 of the 14 datasets with a known value use it; the others are Beem_2021 (167),
  Franke_2025 (169), Bodart_2025a (169.5) and Muldoon_2023 (a range of 168–169.5).
- **Firn correction: from GSFC-FDM v1.2.1** (Medley et al., 2022; on Zenodo, Antarctica on a 12.5 km grid, 1980–2022).
  - It is openly available, and already used for this purpose by Napoleoni et al. (2026). I found no download for
    IMAU-FDM (Veldhuijsen et al., 2023), the alternative.
  - The firn air content (FAC) converts with Eq. 2 of Bodart et al. (2026): `z_f = K rho_i / n_i × FAC = 0.438 × FAC`.
    The 20–25 m of FAC in West Antarctica give 8.8–11 m, which matches the 10 m people use there.
  - Use the long-term mean FAC, and compare with the published values at the sites: EDC 14.6 m, Dome Fuji/B53 15.5 m,
    EDML 13 m, TALDICE 9.5 m, AGAP 12 m, West Antarctica 10 m. Not done yet (an 8 GB download).

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

### 4.3 Chronologies (*Agreed*: the latest published, part of the database)

Every chronology is stored in the database repository with its reference and version, as
`chronologies/<core>/<name>.csv` (depth, age, uncertainty), with `chronologies/cores.json` for the
positions, elevations and depth scales. A published update means: replace the table, rerun, new
harmonisation version. Check each licence before redistributing.

Needed first, because radar reaches these cores (section 1.4). The status is what was checked on
2026-11-01; it must all be read at the source in step 3.

| Core | Chronology | Covers | Status |
|---|---|---|---|
| EDC | AICC2023 (Bouchet et al., 2023) | 813 ka | **Have:** `~/data/AICC2023/` (with AICC2012 side by side) |
| EDML | AICC2023 | 145 ka | **Have** |
| TALDICE | AICC2023 | 492 ka | **Have** |
| Vostok | AICC2023 | 408 ka | **Have** |
| WAIS Divide | WD2014 (Buizert et al., 2015; Sigl et al., 2016) | 68 ka | **Have, to check:** `WDC2014_buizertsigl.tab` has exactly 65,535 rows and stops at 65.5 ka, which looks cut by a spreadsheet limit |
| WAIS Divide ↔ EDC | correspondence WD2014, AICC2012, AICC2023 | | **Have:** `WDC_EDC_chronology.tab` |
| Dome Fuji | DF2021 (Oyabu et al., 2022, CC-BY) | 207 ka | **Have:** `~/data/ice_core_chronologies/DF2021/` (NIPR ADS, 10.17592/001.2022061303). Older part: DFO-2006+AICC2012 |
| South Pole | SP19 (Winski et al., 2019) | 54 ka | To fetch: USAP-DC 601206 |
| Little Dome C | none published | | *Agreed:* use EDC through the radar |
| Byrd | no recent chronology found; Muldoon et al. (2018) use their own Bayesian one | | *Proposed:* do without it, using WD2014 through the radar |
| Aurora Basin North | nothing found; 303 m, and radar only 3.6 km away | | *Proposed:* do without it |

Not reached by radar yet, for later: Skytrain (ST22), RICE (RICE17), Siple Dome, Law Dome, Taylor Dome,
Berkner, Fletcher, James Ross Island.

**Useful:** AICC2023 publishes a correspondence table between AICC2023, AICC2012 and WD2014
(PANGAEA.962566), for the chronology update of 4.5 and for moving between WD2014 and AICC2023.

### 4.4 Horizons reaching several cores, or none

**Several cores (*Proposed*).** Each tie gives an age with its uncertainty. Then:
1. **Consistency check:** is `|a₁ − a₂| ≤ 2 √(σ₁² + σ₂²)`?
2. **Consistent:** the inverse-variance weighted mean. But **the uncertainty is not reduced as if the
   ties were independent**: all AICC2023 cores come from one Bayesian dating, so their errors are
   correlated. Use `σ² = min(σᵢ²) + weighted variance of the ages`, unless the chronology provides
   covariances.
3. **Inconsistent:** never average silently. First suspect the horizon: split it where the matching is
   weakest, and review it. If it holds, take the tie with the smallest uncertainty and the shortest
   connection to the core (fewest crossovers), and keep both ties in `ties`, with the flag
   `core_disagreement`. These cases go in the paper: they measure what the radar and the chronologies
   can resolve.
4. **Mixed chronologies:** bring them to one scale first with the published correspondence (WD2014 ↔
   AICC2023). Otherwise the known offsets between them count as disagreement.

**Reached through crossovers:** the uncertainty adds the depth matching uncertainty along the path,
converted with the local age–depth slope.

**No connection to any core:** keep the original age, with `method = original`, updated to the new
chronology if its tie is known (4.5). A 1D model is outlook.

**Validation (*Proposed*), for the paper and for reviewers:**
- **leave one core out:** date the horizons without a core, and predict its ages;
- **published vs ours:** the appendix table (section 7);
- **no crossings:** harmonised horizons must keep their order everywhere;
- **age–depth order** at every core, and the residuals around loops of crossovers.

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
- **The harmonisation is one rerunnable step** (`anta-harmonise`): its inputs are the line files, the
  chronologies, the dataset parameters of 3.1 and the reviewed links; its outputs are the tables above.
  So a new chronology only needs a rerun, and a new submission gets a proposed horizon and an AntADatabase
  age in the report of its review (the portal's pipeline runs the same matching against the horizons).
- **Radar lines cut into segments:** a new `segment` level under `line`, cut at jumps above 10 km,
  with `Distance` restarting (`metadata-issues.md`, section B).

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

**Parquet (*Agreed*: try it, built from the two views).** One "long" table with one row per depth
value: x, y, depth, horizon, both ages, dataset, line, segment, and the metadata. Sorted by horizon then
line, with row groups, so DuckDB or pyarrow read only what a query needs, even straight from the bucket. It
answers every query from one copy (385 MB measured without the metadata columns), but the netCDF tools can't
read it.

**Where it goes (*Proposed*): alongside the netCDF in `v2/`, not in `v3/`.** A prefix means a breaking
change in the meaning of the data, and this is the same data in another format. Keeping `v3/` free also
leaves room for a later breaking change. The manifest lists it apart, so `sync()` only fetches it on request
(`sync(format="parquet")`), and `Database` uses DuckDB when it is there (an optional extra).

**Gridded products** per horizon: *Agreed*, not in this paper. Transects only.

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

## 8. The plan

Updated 2026-11-11 (first written 2026-11-02). Who: **A** = Antoine, **C** = Claude. The work is on `main`; the
`harmonisation` branch is stale (7 commits behind, none of its own) and can be deleted.

### State on 2026-11-11: the v2 inputs exist

Since 2026-11-02 the plan's input changed: phase 2 no longer starts from the published v1, but from the **v2
build**, compiled from the raw files.
- **v2 build** in `~/data/isochrones/AntADatabase-v2-build`, by `build_v2.py` from anta_json `main` (59cafd3):
  raw_md with the dating fields (`dating method`, `ice core`, `chronology`) and the firn fields per layer,
  missing thickness, bed and surface filled from the Bedmap survey points (`*_SOURCE`, `BEDMAP_POINT_DIST`), the
  Bodart_2021 split traces merged (anta_json #5), the out-of-order lines reordered (#6). `compare_v2.py` explains
  every difference with v1. Shown in `/viewer-dev/` (hidden), its processing page in `/processing/`.
- **Open in anta_json:** #1 (Bodart_2025a, chronology at Dome Fuji), #2 (Leysinger_Vieli_2011, dating per line),
  #3 (Napoleoni_2026, out of v2 until resubmitted), #4 (Winter_2018, 25 lines without a Bedmap thickness), #6 and
  #7 (ordering of lines that turn back on themselves).
- **Dating metadata still missing** (raw_md.json of anta_json main):

  | Missing | Datasets |
  |---|---|
  | `ice core` | Bodart_2025a (#1), Bodart_2025b, Leysinger_Vieli_2011 (#2), Sanderson_2024, Winter_2018 |
  | `chronology` | Bodart_2025a (#1), Leysinger_Vieli_2011 (#2), Muldoon_2023, Siegert_2004 |
  | `firn correction applied` | Beem_2021, Jacobel_2005, Leysinger_Vieli_2011, Siegert_2004, Yan_2025 |
  | `dating method` | Chung_2023 (2 IRHs), Mulvaney_2023 (1 IRH) |

  Ashmore_2020 is modelled, so it needs no core. **Julien is reviewing these fields: wait for his push.**
- **Metadata changes after the review need no recompilation.** The values of the files don't change: set the new
  attributes in the `.nc` files of the dataset (from the new raw_md.json), rebuild the index, compare, then publish.
  To do with a small script (`update_metadata.py`, from raw_md.json to the attributes of the files, with a dry run
  listing the changes), so that it can be rerun after each push.
- **Chronologies:** AICC2023, WD2014, SP19 and DF2021 in hand (section 10).

### Next steps, in order

| # | Step | Who | Needs |
|---|---|---|---|
| N1 | Julien's review of the dating fields, then `update_metadata.py` on the v2 build, reindex | J, then C | His push |
| N2 | ~~Port `explore.py` and `check_depth_correction.py` to the `.nc` files of the v2 build (1.2)~~ Done 2026-11-11 (below) | C | Done |
| N3 | ~~Segments, clusters of crossovers and coincident tracks on the v2 build (2.1)~~ Done 2026-11-11 (below) | C | Done |
| N4 | Tie depths at the cores from the papers (3.1), for the chronology update | A, C | N1 |
| N5 | Chronology update (2.2, level 1 of 4.5): the first table of age changes | C | N1, N4 |
| N6 | Matching, horizons, review, dating, validation (2.3 to 2.7) | C, **A decides** | N3, N5 |
| N7 | ~~Decisions D1 to D5 (8b)~~ Taken 2026-11-11 | A | Done |

N2 and N3 don't depend on the metadata, so they can start while Julien reviews.

### N2 and N3 on the v2 build (2026-11-11)

Scripts in this directory: `load.py` (the lines of the `.nc` files), `crossovers.py` (N3), then `explore.py`
and `check_depth_correction.py` (N2). Output in `~/data/isochrones/harmonisation/AntADatabase-v2-build/`:
`segments.csv`, `runs.csv`, `pairs.parquet`, `pair_depths.parquet`, `summary.md`, `irh_pairs.json`,
`explore.txt`, `depth_correction.txt`. The whole chain takes about a minute.

- **Where the lines meet:** 898 lines, 2,420 segments (cut at gaps over 2 km), over 32 pairs of datasets:
  - 1,674 crossings, plus 25,138 more inside coincident tracks (left out: the track holds them);
  - 307 coincident tracks, 18,485 km: a line of A within 100 m of lines of B for at least 1 km, short
    stretches up to 250 m bridged (Chung's flights weave 6 to 100 m around Mulvaney's). Sampled every km;
  - 1,894 clusters: each coincident track is one, and the crossings of one pair of lines chained within 2 km
    are one. Every comparison is summarised per cluster first, so one shared flight counts once (S3).
  - Largest: Franke x Winter 9,419 km coincident (100 tracks, the same AWI flights), Bodart_2025a x Franke
    1,905 km, Bodart_2021 x Muldoon 1,487 km, Bodart_2025b x Winter 1,454 km, Sanderson x Winter 1,035 km.
    Chung x Mulvaney: 128 km coincident, 15 clusters. They stay different data (decided 2026-11-11).
- **Ice cores (explore.py):** as on 2026-11-01. Within 1 km: EDC, BELDC, Dome Fuji, Vostok, EDML, TALDICE,
  SPICEcore, WDC, Byrd. None near Taylor Dome, Law Dome, Siple Dome, Skytrain, RICE, Fletcher, Berkner, JRI.
- **Nearest-depth pairs (explore.py):** 2,315 pairs of layers seen together. Still fails at depth, as in
  1.3: e.g. Cavitte (now without firn correction) vs Mulvaney is -9 m at 1,073 m but +22 to +35 m below
  1,500 m, and Chung vs Mulvaney picks layers 22 ka apart below 2,000 m. The order-preserving matching (3.3)
  is needed.
- **Depth test (check_depth_correction.py):** 39 layer pairs in at least 3 clusters, with the firn correction
  read from the files. Correlation of measured and predicted offsets +0.51 (-0.18 on 2026-11-01), but the median
  |offset| only drops from 17.6 to 16.1 m: the published parameters still explain little. Left out because the
  value of their firn correction is not known in v2 ("by the authors, value not known", it varies by site):
  Winter, Bodart_2025a, Bodart_2025b, Muldoon; and Beem, Jacobel, Leysinger, Siegert, Yan (not known at all).
  **For Julien's review:** a value per site, where the paper gives one.

### Other v2 items

- **Regions: one `region` argument for ice sheets, IMBIE basins and regions of interest** (*Agreed*
  2026-11-11, details to settle; anta_json issue #8). Today `query(region=...)` only knows
  EAIS/WAIS/APIS, and `query(IMBIE_basin=...)` the IMBIE codes (`Jpp-K`, `Cp-D`...), which few users know.

  **What the user does:**
  ```python
  db.query(region="DC")             # Dome C, by its short name
  db.query(region="Dome C")         # the same, by its name (case-insensitive)
  db.query(region=["WSB", "EAST"])  # several, of any kind: lines in any of them
  db.regions()                      # the table of valid names: short name, name, kind, aliases
  ```
  An unknown name raises an error that lists the valid regions (short and full names).

  **The list, in the package** (`anta_database/data/regions.json`, one entry per region): a short name
  (the key), the full name, the kind, aliases (accepted in queries), and how it is defined. Three kinds:

  | Kind | Examples (short name: name) | Defined by |
  |---|---|---|
  | Ice sheet | `EAST`: East Antarctica (EAIS), `WEST`: West Antarctica (WAIS), `AP`: Antarctic Peninsula (APIS) | The IMBIE regions |
  | Basin | the 18 IMBIE basins, each with a short name (draft below), the code as an alias | The IMBIE polygons |
  | Region of interest | `DC`: Dome C, `DA`: Dome A, `DF`: Dome Fuji, `RB`: Ridge B, `VK`: Vostok, `EDML`: Kohnen/EDML, `TD`: Talos Dome, `SP`: South Pole, `WD`: WAIS Divide, `ASB`: Aurora Subglacial Basin, `WSB`: Wilkes Subglacial Basin, `GSM`: Gamburtsev Subglacial Mountains, `SIR`: Skytrain Ice Rise... | Polygons in the package (below) |

  Draft short names of the IMBIE basins, **to check against the literature** (Rignot et al., 2019;
  Mouginot et al., 2017):

  | Code | Short | Name |
  |---|---|---|
  | K-A | CL | Coats Land (Brunt, Stancomb-Wills) |
  | A-Ap | DML | Dronning Maud Land |
  | Ap-B | EL | Enderby Land |
  | B-C | LAS | Lambert Glacier - Amery Ice Shelf |
  | C-Cp | QML | Queen Mary Land, Wilhelm II Land (Denman, Shackleton) |
  | Cp-D | WL | Wilkes Land (Totten, Moscow University) |
  | D-Dp | GVL | George V Land (Cook, Ninnis, Mertz) |
  | Dp-E | VL | Victoria Land |
  | E-Ep | TAM | Transantarctic Mountains outlets to the Ross Ice Shelf (Byrd, Beardmore) |
  | Jpp-K | FIS | Filchner Ice Shelf (Recovery, Support Force, Foundation) |
  | Ep-F | SC | Siple Coast (Ross ice streams) |
  | F-G | MBL | Marie Byrd Land (Getz) |
  | G-H | ASE | Amundsen Sea Embayment (Pine Island, Thwaites) |
  | H-Hp | BS | Bellingshausen Sea (Abbot, Venable) |
  | J-Jpp | RIS | Ronne Ice Shelf (Rutford, Institute, Evans) |
  | Hp-I | PL | Palmer Land, George VI Ice Shelf |
  | I-Ipp | NAP | northern Antarctic Peninsula |
  | Ipp-J | LIS | Larsen Ice Shelf side of the Peninsula |

  Wilkes Land (`WL`, Cp-D) and the Wilkes Subglacial Basin (`WSB`, mostly in D-Dp) are different places:
  the docs should say so.

  **In the files:** an attribute `regions`, the list of every region the line enters, of all kinds, e.g.
  `["EAST", "WL", "ASB", "DC"]`, assigned like the IMBIE basins today (a line is in a region when one of its
  points is). The current `IMBIE_basins` attribute (`{"Cp-D": "EAIS"}`) is replaced, or kept for v1 only.
  The index gets one regions table (short name, name, kind) and one link table, so that a query by any
  name, of any kind, finds the line.

  **Assigning without recompiling:** a function `assign_regions(nc_files)` (a step of compile, like
  `compute_imbie_basins`, and callable on its own) writes the attribute in place, then the index is
  rebuilt. It runs on the v2 build, and again whenever a region is added to the list.

  **To decide:**
  - How each region of interest is drawn: an outline from the literature where one exists (subglacial
    basins, e.g. from the bed topography), else a circle around the site (which radius: 100 km? per site?),
    or a polygon drawn by hand. The polygons go in the package as GeoParquet, next to the IMBIE basins, with
    their source.
  - The final list of regions of interest, and their short names.
  - Whether `IMBIE_basin=` stays as a deprecated alias of `region=` in the package, for old scripts.
  - The viewer's filters, the download page and the docs use the same short names.

**The order, and why:** the harmonisation decides what the files must hold, so the format is changed once, after it
(section 9). Original layers and ages are never touched, so **a layer that can't be matched with confidence stays its own
horizon with its original age**. The review below 1,500 m can therefore stop at a defensible point without blocking v2.

### Phase 0: published v1 (done)

`v1/` is online (schema 1, 1,014 lines, 248 MB), with packages 0.6.2 and 0.6.3, the site and the portal. Left, none blocks
v2: the checks of `metadata-issues.md`, and copying v1 into `myAntADatabase` (A).

### Phase 1: inputs and baseline

| # | Step | Who | Status |
|---|---|---|---|
| 1.1 | ~~Merge `main` into `harmonisation`; freeze the published v1~~ The input is the v2 build (state above); freeze it for the final run only | C | Replaced |
| 1.2 | Port `explore.py` and `check_depth_correction.py` to the `.nc` files of the v2 build and rerun them: the figures of section 1 predate the 1,014 lines | C | To do (N2) |
| 1.3 | Dataset table `datasets_methods.csv`: check every row, add the missing papers | **A** | Draft (15 of 18 read) |
| 1.4 | Which raw files have travel times | A | To do |
| 1.5 | Chronologies in `chronologies/`, precise core positions, licences | C collects, A checks | Tables fetched (section 10), positions and licences to do |
| 1.6 | Decisions D1 to D5 below | **A** | Open |

### Phase 2: the harmonisation (`anta-harmonise`, rerunnable, writes tables only)

| # | Step | Who |
|---|---|---|
| 2.1 | Segments, clusters of crossovers, coincident tracks (1.1) | C |
| 2.2 | Chronology update only (4.5, level 1): the first table of age changes | C |
| 2.3 | Level 2 offsets, order-preserving matching, horizons (3.2 to 3.4) | C |
| 2.4 | Repeat the depth test on the matched horizons (3.2): **go or no-go for `IRH_DEPTH_H`** | C, **A decides** |
| 2.5 | Horizon review mode in the viewer, local only; then the review, above all below 1,500 m | C builds, **A decides** |
| 2.6 | Depth at the cores, dating, several cores, uncertainties (4.1 to 4.4) | C |
| 2.7 | Validation (4.4): leave one core out, no crossings, loops; table of age changes with causes | C, A checks |

**Gate before phase 3:** A accepts the horizons and the validation. Stable horizon ids are needed from here on (issue T3).

### Phase 3: the new structure

| # | Step | Who |
|---|---|---|
| 3.1 | Compile unchanged, one `.nc` per line as an intermediate. New step `assemble`: the line view, one CF file per dataset | C |
| 3.2 | Layer view, one file per horizon | C |
| 3.3 | Index with lines, segments, horizons, ties. `db.horizon()`, `db.line()`, original or harmonised ages. `query`, `get_files`, `data_generator` and the plotting keep working on the new files | C |
| 3.4 | Viewer, download page and exports on the new files; harmonised colouring | C |
| 3.5 | Portal: `anta-review` assembles and harmonises; a submission's report proposes its horizons | C |
| 3.6 | Parquet, built from the views: compare size and speed, then decide | C |

### Phase 4: release

| # | Step | Who |
|---|---|---|
| 4.1 | Publish to a test prefix, then a clean install, `sync()` and comparison against v1: same values for every original layer | C |
| 4.2 | Docs chapter "Ages and harmonisation" | C, A checks |
| 4.3 | Publish `v2/`, package 0.7.0 or 1.0.0, site. **In this order:** data, package, viewer | **A confirms each** |
| 4.4 | Data DOI for the release, and the manuscript with its generated appendix (section 7) | A, C in the manuscript session |

---

## 8a. Before the harmonisation: the remaining fixes (2026-11-18)

From this plan, `metadata-issues.md`, and the open issues of anta_json and 3d-antadatabase. **Goal:** a frozen v2
build with complete metadata, which phase 2 starts from. Who: **A** = Antoine, **J** = Julien, **C** = Claude.

### 1. Decisions (A, with J), they unblock the rest

| # | Item | Proposal |
|---|---|---|
| F1 | anta_json #12: chronology names. J prefers `null` and a pointer to the paper over unofficial names (`Byrd-BB2001`); Jacobel uses two Byrd chronologies by depth | Keep `null` where no recognised name exists, and say which chronology in the `processing` note. Only the recognised names go in the list of the portal |
| F2 | anta_json #2: Leysinger_Vieli_2011. J's answer: each layer has its core(s); the 3 layers dated at both cores take the middle of the two ages | Close: per layer works. Those 3 layers could become age ranges (min and max of the two datings), as for Napoleoni |
| F3 | anta_json #13: Yan_2025 firn correction. J agrees to "unknown" | `null` already means "not known" in compile: close, or check the paper |
| F4 | anta_json #9: line IDs given by the database (Winter `AWIxxx`) | A `line ID flag` value `assigned`, next to `original`, `not_provided` and `bedmap`, explained in the docs |
| F5 | anta_json #4: Winter_2018, 25 AWI lines of 1997-2002 not filled from Bedmap (26-36 m apart, Bedmap2 at 146 m spacing). J: be more forgiving for old surveys | Allow a distance scaled on the Bedmap point spacing (e.g. half of it) instead of 10 m for every survey, then test the coverage on these 25 lines and the control datasets |
| F6 | Cavitte_2020 depths below the bed (IRH 11 on `ASB_JKB1a_R09Ea`, up to 744 m; IRH 18 and 19 on `VCD_JKB1a_X08a`) | Keep as published, exclude them from the matching (S7), and list them in the processing page |
| F7 | Lines with turns (anta_json #7, Bodart_2025b `W52`, Bodart_2021 `20181022_01_001_025`) | Accept for v2: the values are right, only a small piece is out of order. Solved later by the `segment` level (T9) |

### 2. Metadata to fill in anta_json (J, A)

- [ ] **Line IDs** (#9): Siegert_2004 `005`; Jacobel_2005 `US-ITASE-2001` and `US-ITASE-2002`; Wang_2023 without `IRH_`.
- [ ] **Dating method** of Chung_2023 (2 IRHs) and Mulvaney_2023 (1 IRH).
- [ ] **Chronology** of Jacobel, Siegert, Leysinger-Vieli and Muldoon (one layer), as decided in F1.
- [ ] **Firn correction values per site**, where the papers give one (Winter, Bodart_2025a, Bodart_2025b, Muldoon), for the
      depth test of 2.4.
- [ ] **Beem_2021:** years (2016-2017, done), the project `PPT` that has no line, the wave speed of 167 m/us.

### 3. Data and code fixes (C)

- [ ] **anta_json #10, Cavitte_2020 missing data:** the fix of 2026-11-09 (`fix_cavitte.py`, 108,942 interpolated depths
      removed) covers the 30 HiCARS lines `OIA_JKB2n_*` and IRH 1-20. Find what J still sees (IRH 21-26? MCoRDS
      lines? or v1 in the public viewer), and fix it with the picks of M. Cavitte.
- [ ] **anta_json #6, lines out of order:** the fix is in compile since 2026-11-10 (`order_by_files`). Check the five lines
      in the v2 build, and close.
- [ ] **Line ID flag `assigned`** (F4) in compile and the docs, and the line IDs of #9.
- [ ] **Bedmap distance by survey** (F5), if decided: in `bedmap/reconnect.py`, then rerun the fill.
- [ ] **Check in the v2 build and tick in `metadata-issues.md`:** no layers listed without values (v1 known issue);
      Bodart_2021 OIB 20181022_01 thickness from Bedmap.
- [ ] **`update_metadata.py`:** raw_md.json to the attributes of the `.nc` files, with a dry run, so a change in
      anta_json needs no recompilation. Or rebuild with `build_v2.py` if a fix changes the values (#9, #10, F5).

### 4. Viewer (C, 3d-antadatabase)

- [ ] **#1, Yan_2025:** the profile window is longer than the isochrones. Find which data sets its extent (the surface
      track or the Bedmap fill?) and limit it to the isochrones.
- [ ] **#2:** J likes a different colour for the bed picked in the dataset and the bed taken from Bedmap (`*_SOURCE`). Do
      it, then close.

### 5. Freeze v2 (C, then A)

1. Rebuild v2 from anta_json `main` (22 commits since the build of 2026-11-10, 59cafd3), with `build_v2.py --cpus 3`.
2. `compare_v2.py`: every difference with the last build explained.
3. Refresh the dev viewer and the processing page.
4. Tag the anta_json commit: **the frozen input of phase 2.** Napoleoni_2026 (#3) joins at a rerun when it is
   resubmitted, and so do other submissions (D3).

### State on 2026-11-18

- **Done:**
  - F1 (#12 closed, `null`), F2 (Leysinger_Vieli_2011 ranges of the two cores; #2 closed), F3 (#13 closed);
  - line IDs in `line_ids.json` of anta_json, applied by `anta_database/compiling/line_ids.py`, with the flag
    `assigned` for Winter's `AWI...` lines (#9 closed);
  - Chung and Mulvaney variable files without dating; Beem without `PPT` (#14 asks J);
  - the firn correction per IRH (attribute of `IRH_DEPTH`, `anta_database/firn.py`);
  - `update_metadata.py` run on the v2 build (anta_json 49939d6, tagged `v2-build-2026-11-18`);
  - `Distance` renamed `DIST` in the v2 and Bedmap builds (`rename_dist.py`; the package reads both);
  - `compare_v2.py`: all differences explained;
  - dev viewer data re-exported and uploaded (2026-11-18).
- **Opened:** #15 (high priority, the firn correction of Cavitte_2020), #16 (Cavitte layers below the bed, line
  `ASB_JKB1a_R09Ea` to the coast).
- **Waiting for A:**
  - F5 (#4): the second Bedmap pass, across-track and same institute, proposed in the issue; a first try was reverted;
  - #10: removing the interpolated Cavitte stretches of at least 2 km (found by their 0.1 m steps, 297,708 points
    on 38 lines), proposed in the issue; it also explains #16;
  - publishing the site (`./publish_site.sh`).
- **Done since:** F5 second pass (15 of 24 Winter lines filled, comment on #4). Regions (#8), first version:
  `region=` takes the ice sheets, the IMBIE basins by short name or code, and any of the 20,154 places of
  the SCAR Composite Gazetteer (`anta_database/regions.py`, `data/regions.json`, `data/scar_gazetteer.parquet`),
  a place by a radius per feature type or `radius=`; `db.regions()`, `db.places()`. The index keeps a
  thinned track per line (table `line_tracks`, 1 km); a v1 index reads them from the files. The viewer gets
  `regions.json` (regions with lines) and a Region filter with suggestions, which frames the region.
- **Next for the regions:** check the short names of the basins with Julien; the download page and the
  docs of the site with the same names; Ridge B (not in the gazetteer) and other places to add.

**Not needed before the harmonisation:** the regions (#8, phase 3: J suggests the names of the SCAR Composite
Gazetteer, as Chad Greene's `scarloc`, rather than our own outlines), `Distance` renamed `DIST` (phase 3),
Winter IRH 1 and 5, and the Cavitte firn correction (both settled by the harmonisation itself).

---

## 8c. The harmonisation, as started (2026-11-22)

Replaces the order of phase 2 (section 8) where they differ. Input: the v2 build of 2026-11-21 (its metadata are the
anta_json working tree, pushed and tagged on Monday 2026-11-23: nothing to recompute then). Code: a new module
`anta_database/harmonisation/` (the `anta-harmonise` of section 5), output in `~/data/isochrones/harmonisation/<input>/`.
The portal will run the same code after the compilation of a submission, to propose its horizons and their database age
to the contributor.

**Decided (A, 2026-11-22):**
- **Reference age scale: AICC2023.** The other chronologies (DF2021, SP19, WD2014, and those of the papers) are used to
  compare, and to date what reaches no AICC2023 core, labelled with their scale. Ties to several AICC2023 cores (e.g. EDC
  and EDML) must give the same age: they are compared, and reconciled (H5).
- **Tie depths: our own measure.** The papers rarely give the depth of a layer at the core. We measure it from the radar
  (H3). Where our depth with the authors' chronology does not give their published age back, we look for the information,
  and open an anta_json issue for Julien.
- **Pilot at Dome C**, but with Vostok and Dome Fuji from the start: Winter, Cavitte and Leysinger reach them too. Then
  Dronning Maud Land and Dome Fuji, reconciled with Dome C (above all the layer of about 38 ka seen in every EAIS
  dataset), then West Antarctica on its own (not connected).
- **The depths first, the dating after.** The depth conversion (wave speed, firn) is tested with significant statistics
  before any `IRH_DEPTH_H` (H4).

| # | Step | Output |
|---|---|---|
| H1 | **Core positions**, from the drilling reports (the CSV of the package had errors of 20 to 38 km: Vostok, WDC, TALDICE). **Done 2026-11-22**; with elevation, date of drilling, depth scale | `cores.json`, sources |
| H2 | **Chronology registry**: every table in one format (core, depth, age, uncertainty, scale), AICC2023 first, plus the chronologies of the papers. **Done 2026-11-22** (`harmonisation/chronologies.py`, notes.md H2); open: AICC2023 uncertainty over the last glacial, Vostok-RR2003 is a gas-age scale, depth scales of DF2021 and Vostok | `chronologies/` |
| H3 | **Depth of each published layer at each core it reaches**: all the lines within a radius, a fitted local surface of the layer (Gaussian process: the uncertainty grows with the distance, the length scale from the data), depth below the surface, compared with fractional depth by leaving one line out. Then the age in the authors' chronology (it should give the published age back) and in AICC2023 First pass **2026-11-22** at EDC, Vostok, Dome Fuji (notes.md H3): the depth beats the fractional depth (leave one line out, 2.7 m against 12.3 m) | `ties.csv`, the first table of age changes |
| H4 | **Depth comparison** on the coincident tracks and crossovers (rerun of `crossovers.py` on the build of 2026-11-21): offsets per pair of datasets, and the test of the wave speed and firn conversion on confident pairs | `offsets.csv`, go or no-go for `IRH_DEPTH_H` |
| H5 | **Matching** (3.3, 3.4) at Dome C, Vostok and Dome Fuji, then DML: layers in depth order, the offsets of H4, loops; horizons. Then one age per horizon from all its ties (H3), with the order of the layers kept everywhere | `horizons.csv`, links with their evidence |
| H6 | **Review mode** in the viewer (local): a horizon in one colour across datasets, with its residuals | |
| H7 | **Validation**: leave one core out, order, loops, the table of age changes with their causes | report |

**For Julien on Monday 2026-11-23:** H1 to H4, and a first proposal of H5 for the Dome C pilot.

---

## 8b. Remaining challenges (2026-11-02)

### Science: can change the results

- **S1. The depth corrections may not work.** The test of 3.2 failed (correlation −0.18), though partly on wrong pairs. If it fails on matched horizons too,
  v2 ships without `IRH_DEPTH_H`, and the paper reports the offsets per dataset. The rest of the plan doesn't depend on it.
- **S2. Deep matching.** Below about 1,500 m, layers are as close together as the offsets between datasets. Some links stay a judgement call, and reviewers
  will look there first. Rule proposed above: no confident link, no merge.
- **S3. Agreement isn't independent evidence.** Several datasets took their ages from another by correlation (Bodart_2025a from Franke, Sanderson from Winter,
  Yan from Cavitte), and some share the same radar flights (57 % of Chung's points lie within 50 m of Mulvaney's). Their agreement partly
  reflects that shared origin, so links must be counted per cluster, and the validation must not treat them as separate confirmations.
- **S4. No single age scale across the continent.** The East is on AICC2023; the West is probably a separate network on WD2014. The published WD2014-AICC2023
  table only covers 10 to 57 ka, while WD2014 goes to 67.75 ka. Horizons with no core connection keep older published scales, or model ages (Napoleoni).
  Each age must state its scale, and comparing ages across scales needs care.
- **S5. Dome Fuji.** DF2021 is fetched since 2026-11-10, but only reaches 207 ka. Until then: AICC2012 from Uemura et al. (2018), an older
  scale than the other cores, and a source of false disagreement between cores.
- **S6. Uncertainties.** AICC2023 errors between cores are correlated, and `sigma_offset` depends on the matching. Their definitions differ between papers, and some layers have none
  (Beem_2021, Leysinger_Vieli_2011). The harmonised uncertainty needs one definition, stated in the paper.
- **S7. Doubtful data in the matching.** Open from `metadata-issues.md`: Cavitte depths below the bed, including IRH 11, one of the deep layers that are hard to match (15,311 points on
  one line, up to 744 m below); Napoleoni's jumps; Winter IRH 1 and 5. Proposed: published values stay, these points are excluded from the matching, and the exclusion is listed.
- **S8. Missing parameters.** Jacobel, Leysinger-Vieli and Siegert have no known wave speed or firn correction, Beem no firn correction, and Yan none described. The firn model (GSFC-FDM, 8 GB) isn't downloaded.
  Only level 1 needs them.

### Technical: known problems to design around

- **T1. A real `v2/` with schema 2 would break packages 0.6.0 and 0.6.1.** They read `v2/` and accept schema 2, so they would pass the check and then fail on the new layout, instead of being told to
  upgrade. The stub in `v2/` protects them today by saying schema 3. See D1.
- **T2. Inputs are not all v1.** `AntADatabase` and `AntADatabase-release` still hold the older 956 `.h5` files (`anta-review` refreshes the release copy when it runs), and `explore.py` only reads `.h5`.
  Everything must start from the frozen v1 (step 1.1).
- **T3. Stable horizon ids aren't designed yet.** Section 5 asks for it, but a single new link can merge or split connected components. Proposed: the reviewed links are the source, and a new
  component takes the id of the old horizon it shares most observations with; a merge retires an id, listed in the changelog.
- **T4. Submissions during the work.** A dataset approved now joins v1 and moves the frozen input. See D3.
- **T5. The API on ragged files.** `get_files`, `data_generator`, plotting, viewer export and the portal checks all assume one file per line; each needs porting or a compatibility layer.
  Horizon and age filters also build on a query layer whose older review listed open bugs (ages stored as text, for instance): recheck them in 3.3.
- **T6. The download grows.** Section 6.2 compares 330 to 360 MB with the old 629 MB, but v1 is now 248 MB. Line view plus layer view would be larger than v1: to be measured in 3.2. If it
  matters, `sync()` can fetch the layer view on request, like Parquet.
- **T7. Publishing and citing.**
  - The manuscript brief still has no DOI for the database (line 896). A journal such as ESSD will expect a long-term repository, not only a bucket. A snapshot per release, on Zenodo for example, also needs its licence settled.
  - USAP-DC lists SP19 as CC BY-NC, which could block redistributing that table. If so, the database can link to it and store only the ages derived from it.
- **T8. No automatic checks.** Tests stay out of git, by choice, so the validation of 2.7 should be part of `anta-harmonise` itself and written to its report, so it runs on every rerun.

- **T9. Lines with turns.** One line of a survey can hold several straight profiles joined at turns: `Davis_2026_15` is one 241 km line of that kind, and
  other datasets have the same issue. Not handled for now, as there is no agreed rule for what makes a separate profile. The harmonisation's `segment` level (section 5) is the natural place to solve it.

### Decisions from A (taken 2026-11-11)

| # | Question | Recommendation | Decision |
|---|---|---|---|
| D1 | Address and schema number of the new data | Keep the name `v2/`, with `schema_version` 3 in its manifest (`v{SCHEMA_VERSION}` is split into two constants). Old packages then get the upgrade message. The alternative, using `v3/`, needs no code but drops the agreed name | Keep `v2/`, with `schema_version` 3 in its manifest, as recommended. No `v3/` |
| D2 | What happens to `v1/` after v2 | Frozen and online, so 0.6.2 and 0.6.3 keep working. No new datasets there | `v1/` stays frozen and online |
| D3 | Submissions during the harmonisation | Keep accepting them, published to v1; each new one is added at the next rerun of phase 2. Freeze the input only for the final run | Submissions are received and compiled as now, but **not published to v1**: they wait for v2 and are added to it. `v1/` is never touched again. Until v2, `anta-review accept` and `remove` only publish with `--test` |
| D4 | `BASAL_UNIT` as an undated IRH, and undated layers | Decide now, since the format changes once. Allow horizons without an age in the schema either way; moving `BASAL_UNIT` could still wait, as it isn't needed for the harmonisation, but later it takes another schema change | `BASAL_UNIT` stays as it is: an undated variable, like `ICE_THK`. Given as a layer file, the portal asks no age for it |
| D5 | The common depth reference: 168.5 m/us and GSFC-FDM (3.2b) | Decide only after 2.4: without a go, level 1 isn't built and the 8 GB download isn't needed | Not needed now: it only matters if the test of 2.4 is a go (see the note below the table) |

**D5, the common depth reference:** the datasets convert travel time to depth with different wave speeds
(167 to 169.5 m/us) and firn corrections (none, a constant, or a model). Level 1 of 3.2 would recompute every
depth with one wave speed and one firn model, into `IRH_DEPTH_H`. The test of 2026-11-01 found that these
parameters don't explain the offsets between datasets, so level 1 is only built if the test on matched horizons
(2.4) is a go. Only then are the wave speed and the firn model chosen.

---

## 9. Decisions log

| Date | Decision | Status |
|---|---|---|
| 2026-11-01 | netCDF4 (HDF5 underneath), not Zarr, as the stored format | Agreed |
| 2026-11-01 | `.h5` → `.nc` already for the published v1 | Agreed |
| 2026-11-01 | Bucket: old `v1/` erased, current `v2/` becomes `v1/`; `v2/` = harmonised netCDF | Agreed |
| 2026-11-01 | Compile unchanged, one file per line, as an intermediate | Agreed |
| 2026-11-01 | Line view + layer view + one index | Agreed |
| 2026-11-01 | Try Parquet too, built from the views | Agreed |
| 2026-11-01 | Parquet alongside the netCDF in `v2/`, with `sync(format="parquet")`, for package 0.7.0 or 1.0.0 | Agreed |
| 2026-11-01 | Original IRHs and ages never changed; harmonised horizons added alongside | Agreed |
| 2026-11-01 | No gridded product in this paper | Agreed |
| 2026-11-01 | Latest published chronology of each core, stored in the database | Agreed |
| 2026-11-01 | Date by interpolating the horizon depth at the core, then reading the chronology | Proposed |
| 2026-11-01 | Several cores: consistency check, weighted mean, uncertainty not reduced; disagreement flagged | Proposed |
| 2026-11-01 | `IRH_DEPTH` = as published; `IRH_DEPTH_H` = harmonised, only once the test of 3.2 passes on matched horizons | Agreed naming; the correction is open |
| 2026-11-01 | v1 by direct conversion of the h5 files, no compilation from raw; package 0.6.2 | Agreed |
| 2026-11-01 | Dataset names keep the year of the dataset DOI | Agreed |
| 2026-11-01 | Lines without IDs: `<Dataset>`, or `<Dataset>_<n>` when cut at jumps; reported in the docs and the manuscript | Agreed |
| 2026-11-01 | Attribute `processing` in the files, shown when one dataset is queried | Agreed |
| 2026-11-01 | Apparently duplicated lines (Chung, Mulvaney...) stay as individual lines | Agreed |
| 2026-11-11 | Chung and Mulvaney are different data, never merged; their crossovers count like any other pair's | Agreed |
| 2026-11-01 | Cavitte `ICE_THK`, `BED_ELEV`: +14.6 m removed in v1 | Proposed, Antoine to confirm |
| 2026-11-01 | Harmonise on the current files first, format changed once | Agreed |
| 2026-11-11 | D1: `v2/` with `schema_version` 3 in its manifest | Agreed |
| 2026-11-11 | D2: `v1/` frozen and online | Agreed |
| 2026-11-11 | D3: submissions wait for v2, nothing more is published to `v1/` (accept on hold in `anta-review`) | Agreed |
| 2026-11-11 | D4: `BASAL_UNIT` stays an undated variable | Agreed |
| 2026-11-11 | D5: the common depth reference is chosen only after the test of 2.4 | Agreed |
| 2026-11-11 | One `region` query argument for ice sheets, IMBIE basins (short names) and regions of interest (DC, WSB...); `regions` attribute in the files (section 8, other v2 items) | Agreed, list and outlines to settle |
| 2026-11-22 | Reference age scale AICC2023; other chronologies to compare, or labelled where no AICC2023 core is reached | Agreed |
| 2026-11-22 | Depths of the layers at the cores measured from the radar, not taken from the papers | Agreed |
| 2026-11-22 | Pilot at Dome C with Vostok and Dome Fuji, then DML, then West Antarctica | Agreed |
| 2026-11-22 | Code in a new module `anta_database/harmonisation/`, also run by the portal on submissions | Agreed |

## 10. Chronology files (2026-11-01)

In `~/data/ice_core_chronologies/` (fetched) and `~/data/AICC2023/`.

| Core | Chronology | Source | Range |
|---|---|---|---|
| EDC, EDML, TALDICE, Vostok | AICC2023 | PANGAEA.961017 | |
| WAIS Divide | WD2014 | NOAA `wais-wd2014agemodel.txt` | 0 to 67.75 ka, 3404 m |
| South Pole | SP19 | NOAA `sp19age.txt` (USAP-DC 601206) | 0 to 54.3 ka, 1751 m |
| Dome Fuji | for now AICC2012, from Uemura et al. (2018), `~/data/ice_cores/isotopes/df/uemura_2018/` | NOAA study 23371 | 0 to 716 ka, 3028 m |
| Dome Fuji | DF2021 | NIPR 10.17592/001.2022061303, fetched 2026-11-10: `DF2021/df2021_chronology.txt` (sheet 2 of `S3_DF2021_0-207kyr_BP.xlsx`, ages b1950) | 0 to 207 ka |

**WD2014:** PANGAEA.902577 and `WDC2014_buizertsigl.tab` stop at 65.478 ka after exactly 65,535 rows at
one per year, the limit of a 16-bit row count, not of the method. On those rows they equal the NOAA file,
which goes on to 67.75 ka, as in Buizert et al. (2015). Use the NOAA file.

**WD2014 and AICC2023:** Bouchet et al. (2023) leave WDC out of AICC2023 and recommend timescales tied to
WD2014 over the last 60 kyr. Their table `WDC_EDC_chronology.tab` (395 volcanic ties, 10 to 57 ka) gives
AICC2023 minus WD2014 between -354 and +122 years. **Decided:** date at WDC with WD2014, and give the age on
AICC2023 through this table where it exists. Its WD2014 column is labelled b2k, but equals the BP values of
the NOAA file within 0.5 year.

**Licences to check before the chronologies go in the database:** USAP-DC lists SP19 as CC BY-NC.

**Dome Fuji for now:** Uemura et al. (2018) give the depth on AICC2012, an older timescale than AICC2023 and not the
DF2021 of the plan. DF2021 is now fetched: use it to 207 ka, and AICC2012 from Uemura et al. (2018) only below, marked as such.

**`v2/manifest.json` in the bucket is not a database:** one file with `schema_version` 3 and no data, so that packages 0.6.0 and 0.6.1 are told to upgrade. Before publishing the real v2, settle D1 of section 8b: real data with schema 2 there would be accepted by those packages, which can't read it.
