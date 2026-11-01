# Harmonising the layers, and the data structure that follows

Working document, started 2026-11-01. It records the decisions as they are taken, so that the
manuscript (`../manuscript-brief.md`, and the ESSD draft) can be updated. Nothing here is
implemented yet. **The steps to follow are in section 8, the issues to fix first in
`metadata-issues.md`.** Status of each item:
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
| Dome Fuji | DF2021 (Oyabu et al., 2022, CC-BY) | 207 ka | To fetch: NIPR ADS, 10.17592/001.2022061303. Older part: DFO-2006+AICC2012 |
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

Who: **A** = Antoine, **C** = Claude. Branches: phase 0 is on `main`, since it changes what is online;
the other phases are on `harmonisation`, which merges `main` after phase 0.

### Phase 0: clean up what is published (`main`)

*Agreed:* no compilation from raw. The h5 files of `myAntADatabase` are converted to `.nc` by
`build_v1.py`, with the encoding of `anta_database/netcdf.py` (float32 values, zlib 4 + shuffle, larger chunks).

| # | Step | Status |
|---|---|---|
| 0.1 | Issues of `metadata-issues.md` | Answered; some checks left, none blocks v1 |
| 0.2 | Line IDs, `processing`, units, compile settings | Done (not committed) |
| 0.3 | `.h5` → `.nc` in code and docs; `SCHEMA_VERSION = 1`; default URLs to `v1/` | Done (not committed) |
| 0.4 | Build in `~/data/isochrones/AntADatabase-v1-build`: **248 MB, 1,014 lines** (was 630 MB, 956) | Done, checked value by value (at most 0.1 mm), 18 datasets, publish dry-run OK |
| 0.5 | Antoine checks the build, then: commit, tag 0.6.2, publish `v1/`, exports to `web/`, site, portal, delete `v2/` and `test/` | Waits for A |

**Yank (optional):** older packages would break on the new data. Marking 0.5.0–0.6.1 as "yanked" on PyPI makes
`pip install` skip them. Antoine considers the crash unimportant, so it can be skipped.

### Phase 1: collect what the methods need

| # | Step | Who |
|---|---|---|
| 1.1 | Per dataset: wave speed, firn correction, dating method, core, chronology, tie. **Draft done:** `datasets_methods.csv`, with quotes (15 of 18 from the papers) | **A checks** every row, adds the missing papers |
| 1.2 | Published travel times per dataset: which raw files have them | A (has the raw files) |
| 1.3 | Choose the common wave speed and the firn model | A decides, C proposes |
| 1.4 | Chronologies and precise core positions (4.3), and their licences | C collects, A checks |

### Phase 2: harmonise

| # | Step | Who |
|---|---|---|
| 2.1 | Segments; clusters of crossovers; coincident tracks (1.1) | C |
| 2.2 | Level 1 depth correction, and its test against the measured offsets (3.2) | C |
| 2.3 | Chronology update only (4.5, level 1): the first table of age changes | C |
| 2.4 | Level 2 offsets, order-preserving matching, horizons (3.2–3.4) | C |
| 2.5 | Horizon review mode in the viewer, and **the review of the horizons, above all below 1,500 m** | C builds, **A decides** |
| 2.6 | Depth at the cores, dating, several cores, uncertainties (4.1–4.4) | C |
| 2.7 | Validation (4.4), and the table of age changes with its causes | C, A checks |

Where it can go wrong: the code is days of work, but **the correctness of the matching isn't a coding
problem**. Below 1,500 m, the layers are as close together as the offsets between datasets; some links will
stay a judgement call, and those are yours. Reviewers will look there first.

### Phase 3: the new format (`v2/`, schema 2, package 0.7.0 or 1.0.0)

| # | Step | Who |
|---|---|---|
| 3.1 | **Compile unchanged**: still one `.nc` per line, as an intermediate. A new step `assemble` builds the line view (one file per dataset), adding CF attributes (`Conventions`, `featureType`, `grid_mapping` with EPSG:3031; no spaces in attribute names) | C |
| 3.2 | Layer view (6.2), generated | C |
| 3.3 | Index: line, segment, horizon, ties; `db.horizon()`, `db.line()`, original or harmonised ages | C |
| 3.4 | Viewer, download page and portal on the new files; a harmonised colouring in the viewer | C |
| 3.5 | Parquet, built from the views; compare size and speed with netCDF, then decide | C |
| 3.6 | Publish to `v2/` (test prefix first), `anta-harmonise` for new submissions | C, A confirms |

### Phase 4: documentation and manuscript

Section 7: a chapter "Ages and harmonisation" in the docs, and the changes to the manuscript, with the
appendix generated by script. C, in the manuscript session; A decides the content.

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
| 2026-11-01 | Cavitte `ICE_THK`, `BED_ELEV`: +14.6 m removed in v1 | Proposed, Antoine to confirm |
| 2026-11-01 | Harmonise on the current files first, format changed once | Agreed |

## 10. Chronology files (2026-11-01)

In `~/data/ice_core_chronologies/` (fetched) and `~/data/AICC2023/`.

| Core | Chronology | Source | Range |
|---|---|---|---|
| EDC, EDML, TALDICE, Vostok | AICC2023 | PANGAEA.961017 | |
| WAIS Divide | WD2014 | NOAA `wais-wd2014agemodel.txt` | 0 to 67.75 ka, 3404 m |
| South Pole | SP19 | NOAA `sp19age.txt` (USAP-DC 601206) | 0 to 54.3 ka, 1751 m |
| Dome Fuji | for now AICC2012, from Uemura et al. (2018), `~/data/ice_cores/isotopes/df/uemura_2018/` | NOAA study 23371 | 0 to 716 ka, 3028 m |
| Dome Fuji, later | DF2021 | NIPR 10.17592/001.2022061303: **not fetched**, site in maintenance on 2026-11-01, no other copy found | 0 to 207 ka |

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
DF2021 of the plan: replace it when NIPR is back, and mark the ages dated with it until then.

**`v2/manifest.json` in the bucket is not a database:** one file with `schema_version` 3 and no data, so that packages 0.6.0 and 0.6.1 are told to upgrade. Delete it before publishing the real v2 (`anta-publish` refuses a prefix holding another schema).
