"""The processing page of the website (/processing/, not linked and not indexed for now): what the
compilation of v2 does to each dataset, from the build itself, so it follows the data.

Reads, in BUILD (build_v2.py): the index (the processing of each dataset, as in the files), the
raw_md.json of each dataset (platform, dating), bedmap_fill.csv (the reconnection to Bedmap) and
build_info.json (the commit of the metadata); and datasets_methods.csv of this directory, what the
articles say. Writes web/processing/index.html. Run by build_v2.py, or alone:

    python docs/harmonisation/processing_page.py [BUILD]
"""
import csv
import json
import sqlite3
import sys
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAGE = REPO / "web" / "processing" / "index.html"
BUILD = Path.home() / "data" / "isochrones" / "AntADatabase-v2-build"
METHODS = HERE / "datasets_methods.csv"
ANTA_JSON = "https://github.com/antoinehermant/anta_json"

# The steps, in the order of the compilation (compile.py, then reconnect.py)
STEPS = """
<li><strong>Raw files, as published.</strong> Each dataset is compiled again from the files of its
authors, with its metadata from <a href="{anta_json}">anta_json</a> (commit <code>{commit}</code>). Corrected
files given by the authors replace the published ones where listed below.</li>
<li><strong>Depths.</strong> Travel times are converted to depths with the wave speed of the article, and the
firn correction is added, only where the published depths have none (Cavitte_2020, Wang_2023). Elsewhere the
depths stay as published, with the firn correction of their authors (table below).</li>
<li><strong>Lines.</strong> A line is one file of the database, named by its line ID. Lines are never cut. The
points of a line are put in order along it where its layers were published in separate files. A line only lists
the layers it has values of.</li>
<li><strong>Ice thickness, bed and surface elevation.</strong> Where a raw file gives two of the three, the third
is computed (S = B + T). Where a line has none, they are taken from the Bedmap survey line of the same radar
traces: the nearest Bedmap survey point, never the gridded Bedmap products (rules below).</li>
<li><strong>Where each value comes from.</strong> <code>ICE_THK_SOURCE</code>, <code>BED_ELEV_SOURCE</code> and
<code>SURF_ELEV_SOURCE</code>, per point: 0 the published dataset, 1, 2, 3 a Bedmap1, Bedmap2 or Bedmap3 survey
point, 4 computed from the two other published variables; missing where there is no value.
<code>BEDMAP_POINT_DIST</code>: the distance to that Bedmap point.</li>
<li><strong>Platform and dating.</strong> Each line says whether its radar was airborne or ground-based
(<code>platform</code>), and each layer how it was dated: the attributes <code>dating method</code> (ice-core or
modelled), <code>ice core</code> and <code>chronology</code> of <code>IRH_AGE</code>, one value per layer. Both can be
queried:
<code>db.query(platform="ground-based", dating_method="ice-core")</code>.</li>
"""

RECONNECTION = """
<ul>
<li>Only Bedmap surveys of the same platform: airborne lines with the airborne surveys, ground-based lines with the
ground surveys (and the seismic soundings). The radar on the ground traces other ice than an aircraft flying over
it.</li>
<li>For each point, the nearest point of the Bedmap survey within 50 m. One survey per line, chosen by the most
points matched (the newest generation when two are close): values are never mixed from two surveys.</li>
<li>The survey is only taken if its points are within {max_median:g} m of those of the line, on median: the same
traces are 0 to 7 m apart, other flights over the same place 25 to 42 m. {exceptions}</li>
<li>Published values are never replaced. A line with nothing missing is not changed.</li>
</ul>
"""


def html_table(header: list, rows: list) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def joined(values) -> str:
    values = [v for v in (values if isinstance(values, list) else [values]) if v not in (None, "")]
    return ", ".join(dict.fromkeys(str(v) for v in values))


def metadata(build: Path, dataset: str) -> dict:
    """Platform and dating of a dataset, from its raw_md.json."""
    rows = json.loads((build / dataset / "raw_md.json").read_text(encoding="utf-8"))
    layers = [r for r in rows if isinstance(r.get("IRH"), int)]
    out = {"platform": joined(rows[0].get("platform"))}
    for field in ("dating method", "ice core", "chronology"):
        out[field] = joined([r.get(field) for r in layers])
    return out


def dataset_rows(build: Path) -> list:
    with sqlite3.connect(f"file:{build / 'AntADatabase.db'}?mode=ro", uri=True) as conn:
        sources = conn.execute("SELECT name, citation, DOI_article, processing FROM sources ORDER BY name").fetchall()
    methods = {r["dataset"]: r for r in csv.DictReader(METHODS.open(encoding="utf-8"))}
    rows = []
    for name, citation, doi, processing in sources:
        if name.startswith("BEDMAP") or not (build / name).is_dir():  # not in the build (any more)
            continue
        md = metadata(build, name)
        paper = methods.get(name, {})
        firn = paper.get("firn_correction_m") or "not stated"
        if paper.get("firn_in_published_depths"):
            firn += f" ({paper['firn_in_published_depths']})"
        dating = " ".join(
            f"{label} {escape(value)}."
            for label, value in (("Dating:", md["dating method"]), ("Core:", md["ice core"]), ("Chronology:", md["chronology"]))
            if value
        )
        rows.append([
            f'<a href="{escape(doi)}">{escape(citation)}</a><br><code>{escape(name)}</code>',
            escape(md["platform"]),
            escape(paper.get("wave_speed_m_per_us", "")),
            escape(firn),
            dating or "<em>not given yet</em>",
            escape(processing or "As published."),
        ])
    return rows


def reconnection_rows(build: Path) -> list:
    report = build / "bedmap_fill.csv"
    if not report.exists():
        return []
    df = pd.read_csv(report)
    if "status" not in df.columns:  # a build before the rules of 2026-11-10
        return []
    rows = []
    for name, d in df.groupby("dataset"):
        filled = d[d.status == "filled"]
        refused = d[d.status != "filled"]
        surveys = filled.groupby(["generation", "survey"]).size().sort_values(ascending=False)
        rows.append([
            f"<code>{escape(name)}</code>",
            f"{len(filled)} of {len(d)}",
            escape(", ".join(f"{g} {s} ({n})" for (g, s), n in surveys.head(4).items())
                   + (", ..." if len(surveys) > 4 else "")) or "-",
            f"{filled.median_dist.median():.1f} m" if len(filled) else "-",
            ", ".join(f"{v} {int(filled[v].sum()):,}" for v in ("ICE_THK", "BED_ELEV", "SURF_ELEV")
                      if v in filled and filled[v].sum() > 0) or "-",
            escape("; ".join(f"{s} ({n})" for s, n in refused.status.value_counts().items())) or "-",
        ])
    return rows


def bedmap_dropped() -> str:
    """The Bedmap values found wrong and left out (anta_database.bedmap.convert)."""
    from anta_database.bedmap.convert import DROPPED, DROPPED_PARTS

    items = [f"<li><code>{escape(survey)}</code>: {escape(reason)}</li>" for survey, (_, reason) in DROPPED.items()]
    items += [f"<li><code>{escape(survey)}</code>: {escape(rule['reason'])}</li>" for survey, rule in DROPPED_PARTS.items()]
    return (
        "<p>Each Bedmap survey was compared with the Bedmap3 grid. Values found wrong are left out of the Bedmap "
        "lines, so they never reach the isochrone lines, and the reason is in the attribute <code>processing</code> "
        "of their files:</p><ul>" + "".join(items) + "</ul>"
        "<p>Where a Bedmap line has no surface (as these NASA surveys now), the 3D viewer draws it at the Bedmap3 "
        "surface.</p><p>Known issue, kept as published for now: <code>BAS_2007_AGAP_AIR_BM2</code>, line 69, whose "
        "last 240 km lie on the Ross Ice Shelf with the values of the plateau (surface 3,400 m): wrong positions, to "
        "report to the Bedmap team. No isochrone line is near it.</p>"
    )


def page(build: Path) -> str:
    from anta_database.bedmap.reconnect import MAX_MEDIAN

    sys.path.insert(0, str(HERE))
    from build_v2 import DATASETS, MAX_MEDIAN as EXCEPTIONS

    info = json.loads((build / "build_info.json").read_text()) if (build / "build_info.json").exists() else {}
    commit = info.get("anta_json", {}).get("commit", "")[:10] or "not recorded"
    exceptions = " ".join(
        f"Exception: {escape(d)}, {m:g} m, whose flights Bedmap1 has in short pieces only (so its fill has gaps)."
        for d, m in EXCEPTIONS.items()
    )
    reconnection = reconnection_rows(build)
    excluded = sorted({p.name for p in (REPO.parent / "anta_json").glob("*_*/") if not p.name.startswith("BEDMAP")} - set(DATASETS))
    parts = [
        "<h2>How the data is processed</h2>",
        '<p class="draft">Draft for version 2 of the database, not published yet. Built on '
        f"{date.today().isoformat()} from the build of version 2.</p>",
        "<h3>Steps</h3>",
        "<ol>" + STEPS.format(anta_json=ANTA_JSON, commit=escape(commit)) + "</ol>",
        "<h3>Each dataset</h3>",
        "<p>The wave speed (m/µs) and firn correction (m) of the articles, how the layers were dated, and what the "
        "compilation does to the raw files.</p>",
        html_table(["Dataset", "Platform", "Wave speed", "Firn correction", "Dating", "Compilation"], dataset_rows(build)),
        "<h3>Ice thickness, bed and surface from Bedmap</h3>",
        RECONNECTION.format(max_median=MAX_MEDIAN, exceptions=exceptions),
        html_table(["Dataset", "Lines filled", "Bedmap survey (lines)", "Distance (median)", "Points filled", "Not filled"],
                   reconnection)
        if reconnection else "<p><em>The table comes with the next build of version 2.</em></p>",
        "<h3>Bedmap values left out</h3>",
        bedmap_dropped(),
    ]
    if excluded:
        parts.append("<h3>Not in version 2 yet</h3><p>" + ", ".join(f"<code>{escape(e)}</code>" for e in excluded)
                     + ": under review with their authors.</p>")
    return TEMPLATE.replace("{{content}}", "\n      ".join(parts))


TEMPLATE = """<!doctype html>
<html lang="en">
  <!-- Generated by docs/harmonisation/processing_page.py: do not edit -->
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="color-scheme" content="light dark" />
    <meta name="robots" content="noindex, nofollow" />
    <title>Processing · AntADatabase</title>
    <link rel="icon" type="image/png" sizes="192x192" href="/favicon.png" />
    <link rel="icon" type="image/png" sizes="48x48" href="/favicon-48.png" />
    <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
    <meta name="theme-color" content="#2a6aa8" />
    <link rel="stylesheet" href="/src/processing.css" />
    <script>
      // Theme before the first paint, as in src/theme.ts
      const theme = localStorage.getItem("theme");
      const dark = theme !== "Light" && (theme !== "Auto" || matchMedia("(prefers-color-scheme: dark)").matches);
      document.documentElement.dataset.theme = dark ? "dark" : "light";
    </script>
  </head>
  <body>
    <header id="top">
      <div class="wrap">
        <h1><a href="/"><img class="logo" src="/logo.png" alt="" width="34" height="32" />AntADatabase</a></h1>
        <nav>
          <a href="/viewer/">3D viewer</a>
          <a href="https://docs.antadatabase.com/intro.html">Python database</a>
          <a href="/download/">Download</a>
          <a href="/submit/">Submit</a>
          <a href="/contact/">Contact</a>
        </nav>
        <button id="theme-toggle" class="theme-toggle" type="button"></button>
      </div>
    </header>

    <main class="wrap">
      <p class="back"><a href="/">← Home</a></p>
      {{content}}
    </main>
    <footer class="wrap muted">
      Developed as part of the CHARIBDIS project (Swiss National Science Foundation grant no. 211542), University of
      Bern. Source code on <a href="https://github.com/antoinehermant/anta_database" target="_blank" rel="noopener">GitHub</a>
      (MIT licence). Contact: <a href="mailto:antoine.hermant@unibe.ch">antoine.hermant@unibe.ch</a>. <a href="/privacy/">Privacy</a>.
    </footer>
    <script type="module" src="/src/page.ts"></script>
  </body>
</html>
"""


def main(build: Path = BUILD) -> None:
    PAGE.parent.mkdir(exist_ok=True)
    PAGE.write_text(page(build), encoding="utf-8")
    print(f"Written {PAGE}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else BUILD)
