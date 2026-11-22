"""The chronologies of the ice cores (H2): every published depth-age table in one format.

One table per chronology and core: depth (m, the depth scale of the core's log), ice age (ka before 1950)
and its uncertainty (ka, 1 sigma; NaN where not published). Ages given before 2000 (b2k) or in years are
converted; uncertainties given at 2 sigma are halved. Gas ages are not used: the layers are in the ice.

The sources are the files as published (data centres), read where they were downloaded (SOURCES, paths
under the data directory, ~/data by default): the database does not redistribute them before their
licences are checked (SP19 is CC BY-NC). `build(out)` writes one CSV per table and registry.json.

AICC2023 is the reference age scale of the harmonisation (plan.md, 8c). The others date the layers as the
papers did (raw_md.json, "chronology"), to check that our depth at the core gives the published age
back (H3), and date what no AICC2023 core reaches.

    python -m anta_database.harmonisation.chronologies [output dir] [--data ~/data]
"""
import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/data")
OUTPUT = os.path.expanduser("~/data/isochrones/harmonisation/chronologies")


@dataclass
class Chronology:
    name: str  # as in chronologies.json
    core: str  # as in ice_cores.csv
    file: str  # under the data directory
    reference: str
    doi: str
    source: str  # data centre and dataset
    age_reference: str  # of the file: "BP1950" or "b2k"
    uncertainty: str  # of the file, as published
    note: str = ""
    usable: bool = True  # False: an ice age can't be taken from it
    read: Optional[Callable[[str], pd.DataFrame]] = field(default=None, repr=False)


def _pangaea(path: str) -> pd.DataFrame:
    """A PANGAEA tab file: the table after the metadata (between /* and */), if any."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    end = next((i for i, l in enumerate(lines) if l.lstrip("#").strip() == "*/"), -1)
    from io import StringIO

    return pd.read_csv(StringIO("\n".join(lines[end + 1:])), sep="\t")


def _aicc(version: str) -> Callable[[str], pd.DataFrame]:
    def read(path: str) -> pd.DataFrame:
        df = _pangaea(path)
        age = df[f"Ice age [ka BP] ({version})"]
        unc = df.get(f"Ice age unc [±] ({version})", pd.Series(np.nan, index=df.index))
        return pd.DataFrame({"depth": df["Depth ice/snow [m]"], "age": age, "age_unc": unc})
    return read


def _table(path: str, start: str, encoding: str = "latin-1", sep: str = r"\s+", names=None) -> pd.DataFrame:
    """A text file with a free header: the table from the line starting with `start` (its header, replaced
    by names if given)."""
    with open(path, encoding=encoding, newline="") as f:
        lines = re.split(r"\r\n|\r|\n", f.read())
    i = next(i for i, l in enumerate(lines) if l.strip().startswith(start))
    from io import StringIO

    if names:
        return pd.read_csv(StringIO("\n".join(lines[i + 1:])), sep=sep, engine="python", header=None, names=names)
    return pd.read_csv(StringIO("\n".join(lines[i:])), sep=sep, engine="python")


def _noaa(path: str) -> pd.DataFrame:
    """A NOAA template file: the table after the # lines."""
    return pd.read_csv(path, sep="\t", comment="#")


def _edc3(path: str) -> pd.DataFrame:
    df = _table(path, "Bag", names=["bag", "depth", "ice", "air"])  # header "Bag Depth Ice Age Air Age"
    return pd.DataFrame({"depth": df.depth, "age": df.ice / 1000, "age_unc": np.nan})


def _wd2014(path: str) -> pd.DataFrame:
    df = _noaa(path)
    # Annual layer counting above 2850 m, CH4 synchronisation below: the two parts of the uncertainty (2 sigma)
    unc = np.hypot(df.age_ice_ALC_unc.fillna(0), df.age_ice_SYNC_unc.fillna(0)) / 2
    return pd.DataFrame({"depth": df.depth_m, "age": df.age_ice_calBP / 1000, "age_unc": unc / 1000})


def _sp19(path: str) -> pd.DataFrame:
    df = _noaa(path)
    return pd.DataFrame({"depth": df.depth_m, "age": df.age_SP19 / 1000, "age_unc": df["age+-Tot"] / 1000})


def _df2021(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#")
    return pd.DataFrame({"depth": df["Depth (top, m)"], "age": df["Ice age (top, yr b1950)"] / 1000,
                         "age_unc": df["Ice age uncert. (yr, 2σ)"] / 2000})


def _dfo2006(path: str) -> pd.DataFrame:
    df = _table(path, "Depth     IceAge")
    return pd.DataFrame({"depth": df.Depth, "age": df.IceAge / 1000, "age_unc": np.nan})


def _gt4(path: str) -> pd.DataFrame:
    df = _noaa(path)
    return pd.DataFrame({"depth": df.depth_m, "age": df.ice_ageBP / 1000, "age_unc": np.nan})


def _byrd2001(path: str) -> pd.DataFrame:
    df = _table(path, "Depth\td18O", sep="\t").iloc[1:]  # the second line: units
    # Byrd: columns 7 to 10 (depth, d18O, ice age on GISP2 in yr BP1950, ice age on GRIP in yr b1989)
    depth, age = pd.to_numeric(df.iloc[:, 6], errors="coerce"), pd.to_numeric(df.iloc[:, 8], errors="coerce")
    ok = depth.notna() & age.notna()
    return pd.DataFrame({"depth": depth[ok].values, "age": age[ok].values / 1000, "age_unc": np.nan})


AICC = "Bouchet et al. (2023), Climate of the Past 19"
AICC_DOI = "10.5194/cp-19-2257-2023"
AICC12 = "Bazin et al. (2013); Veres et al. (2013), Climate of the Past 9"

SOURCES: List[Chronology] = [
    *[Chronology("AICC2023", core, f"AICC2023/{file}_chronology.tab", AICC, AICC_DOI,
                 "PANGAEA.961017", "BP1950", "1 sigma", read=_aicc("AICC2023"))
      for core, file in [("EDC", "EDC"), ("EDML", "EDML"), ("TALDICE", "TALDICE"), ("Vostok", "VOSTOK")]],
    *[Chronology("AICC2012", core, f"AICC2023/{file}_chronology.tab", AICC12, "10.5194/cp-9-1715-2013",
                 "PANGAEA.961017 (the AICC2012 column of AICC2023)", "BP1950", "not given in these files",
                 note="Uncertainties of AICC2012 not in the AICC2023 files", read=_aicc("AICC2012"))
      for core, file in [("EDC", "EDC"), ("EDML", "EDML"), ("TALDICE", "TALDICE"), ("Vostok", "VOSTOK")]],
    Chronology("EDC3", "EDC", "ice_core_chronologies/EDC3/edc3-timescale.txt", "Parrenin et al. (2007), Climate of the Past 3",
               "10.5194/cp-3-485-2007", "NOAA WDS-Paleo 2007-083", "BP1950", "not given", read=_edc3),
    Chronology("WD2014", "WDC", "ice_core_chronologies/WD2014/wais-wd2014agemodel.txt",
               "Buizert et al. (2015); Sigl et al. (2016), Climate of the Past", "10.5194/cp-11-153-2015",
               "NOAA (wais-wd2014agemodel.txt)", "BP1950", "2 sigma (layer counting and CH4 synchronisation, combined)",
               note="PANGAEA.902577 stops at 65.5 ka (65,535 rows); the NOAA file goes to 67.75 ka", read=_wd2014),
    Chronology("SP19", "SPICEcore", "ice_core_chronologies/SP19/sp19age.txt", "Winski et al. (2019), Climate of the Past 15",
               "10.5194/cp-15-1793-2019", "USAP-DC 601206 (NOAA copy)", "BP1950",
               "total, sigma level not stated (taken as 2 sigma, as WD2014 to which it is tied)",
               note="Licence CC BY-NC", read=_sp19),
    Chronology("DF2021", "Dome Fuji", "ice_core_chronologies/DF2021/df2021_chronology.txt",
               "Oyabu et al. (2022), Quaternary Science Reviews 294", "10.1016/j.quascirev.2022.107754",
               "NIPR ADS 10.17592/001.2022061303 (sheet 2 of S3)", "BP1950", "2 sigma", read=_df2021),
    Chronology("DFO-2006", "Dome Fuji", "ice_core_chronologies/DFO2006/dfo-2006-timescale-340ka.txt",
               "Kawamura et al. (2007), Nature 448", "10.1038/nature06015", "NOAA WDS-Paleo 2007-074", "b2k",
               "not given", note="DF1 depth; unchanged from Watanabe et al. (2003) above 41.5 ka", read=_dfo2006),
    Chronology("GT4", "Vostok", "ice_core_chronologies/GT4/gt4nat-noaa.txt", "Petit et al. (1999), Nature 399",
               "10.1038/20859", "NOAA study 2453", "BP1950", "not given",
               note="BP, reference year not stated (conventionally 1950)", read=_gt4),
    Chronology("Byrd-BB2001", "Byrd", "ice_core_chronologies/Byrd/iso_blunier01.txt",
               "Blunier and Brook (2001), Science 291", "10.1126/science.291.5501.109", "NOAA study 2450", "BP1950",
               "not given", note="CH4 synchronisation to GISP2, 940 to 2142 m (10 to 90 ka) only; the GISP2 column",
               read=_byrd2001),
    Chronology("Vostok-RR2003", "Vostok", "ice_core_chronologies/Ruddiman2003/vostok_methane_age.txt",
               "Ruddiman and Raymo (2003), Quaternary Science Reviews 22", "10.1016/S0277-3791(02)00082-8",
               "NOAA WDS-Paleo 2004-016", "BP1950", "not given",
               note="A gas-age scale (CH4 tuned to insolation): the file gives no ice age", usable=False),
]


def sources(data: str = DATA) -> List[Chronology]:
    return [s for s in SOURCES if os.path.exists(os.path.join(data, s.file))]


def read(source: Chronology, data: str = DATA) -> pd.DataFrame:
    """The table of a source: depth (m), age (ka BP1950), age_unc (ka, 1 sigma), sorted by depth, checked."""
    df = source.read(os.path.join(data, source.file)).apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["depth", "age"]).sort_values("depth").drop_duplicates("depth").reset_index(drop=True)
    if source.age_reference == "b2k":
        df["age"] -= 0.05
    if not (np.diff(df.age) > 0).all():
        raise ValueError(f"{source.name} {source.core}: the age does not increase with depth")
    return df


@lru_cache(maxsize=None)
def table(name: str, core: str, root: str = OUTPUT) -> pd.DataFrame:
    """A built table (build): depth, age, age_unc."""
    return pd.read_csv(os.path.join(root, core, f"{name}.csv"))


def age_at(name: str, core: str, depth, root: str = OUTPUT) -> Tuple[np.ndarray, np.ndarray]:
    """Age and its uncertainty (ka) at depths (m) of a core, linear between the rows; NaN outside the table."""
    t = table(name, core, root)
    d = np.asarray(depth, float)
    kw = dict(left=np.nan, right=np.nan)
    return np.interp(d, t.depth, t.age, **kw), np.interp(d, t.depth, t.age_unc, **kw)


def depth_at(name: str, core: str, age, root: str = OUTPUT) -> np.ndarray:
    """Depth (m) of ages (ka) at a core: the inverse of age_at."""
    t = table(name, core, root)
    return np.interp(np.asarray(age, float), t.age, t.depth, left=np.nan, right=np.nan)


WD2014_AICC2023 = "AICC2023/WDC_EDC_chronology.tab"  # Bouchet et al. (2023): 395 volcanic ties WDC-EDC, 10 to 57 ka


def wd2014_to_aicc2023(age, data: str = DATA) -> np.ndarray:
    """AICC2023 age (ka BP1950) of WD2014 ages (ka BP1950), through the volcanic ties of Bouchet et al. (2023),
    linear between them; NaN outside 10 to 57 ka. Their WD2014 column says b2k but equals the BP1950 of the NOAA
    file within 0.5 year (plan.md, section 10), so both columns are taken as BP1950."""
    t = _pangaea(os.path.join(data, WD2014_AICC2023))
    wd, aicc = t["Ice age [ka b2k] (WD2014)"].values, t["Ice age [ka b2k] (AICC2023)"].values
    o = np.argsort(wd)
    return np.interp(np.asarray(age, float), wd[o], aicc[o], left=np.nan, right=np.nan)


def build(out: str = OUTPUT, data: str = DATA) -> pd.DataFrame:
    """Write <out>/<core>/<name>.csv for every usable source found, and registry.json; the summary."""
    rows = []
    for s in sources(data):
        meta = {k: v for k, v in asdict(s).items() if k != "read"}
        if not s.usable:
            rows.append({**meta, "rows": 0})
            continue
        df = read(s, data)
        os.makedirs(os.path.join(out, s.core), exist_ok=True)
        df.to_csv(os.path.join(out, s.core, f"{s.name}.csv"), index=False, float_format="%.6g")
        rows.append({**meta, "rows": len(df), "depth_min": df.depth.min(), "depth_max": df.depth.max(),
                     "age_min_ka": df.age.min(), "age_max_ka": df.age.max(),
                     "has_uncertainty": bool(df.age_unc.notna().any())})
    missing = [f"{s.name} {s.core}" for s in SOURCES if s not in sources(data)]
    with open(os.path.join(out, "registry.json"), "w") as f:
        json.dump({"age": "ka before 1950", "uncertainty": "ka, 1 sigma", "chronologies": rows, "missing": missing},
                  f, indent=1, default=float)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", nargs="?", default=OUTPUT)
    p.add_argument("--data", default=DATA)
    a = p.parse_args(argv)
    summary = build(a.out, a.data)
    cols = ["name", "core", "rows", "depth_min", "depth_max", "age_min_ka", "age_max_ka", "has_uncertainty"]
    print(summary[cols].to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()
