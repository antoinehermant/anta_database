/**
 * Download page: choose datasets and download them in one zip file.
 *
 * The descriptions come from the catalogue exported with the viewer data
 * (python -m anta_database.web.catalog), the files from the manifest of the
 * published database, so the download always matches the bucket. The cards and
 * the zip are in datasets.ts, shared with the download frame of a private preview.
 */
import { type Catalog, DATA_ROOT, fetchData } from "./sitedata";
import { type Dataset, datasetCard, element, formatSize, saveZip } from "./datasets";
import { themeButton } from "./theme";

// Published database (bucket prefix of manifest.json), as in anta_database.cloud.sync
const DB_URL = (
  import.meta.env.VITE_DB_URL ||
  "https://s3.pub1.infomaniak.cloud/object/v1/AUTH_2030d6d085ec4b16a7c2320f0bb9427d/antadatabase/v2"
).replace(/\/$/, "");
const ROOT_DIR = "AntADatabase";
const DB_FILE = "AntADatabase.db";
const MANIFEST_FILE = "manifest.json";

interface Manifest {
  db_version?: string;
  created?: string;
  files: Record<string, { sha256: string; size: number }>;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const list = $("datasets");
const selectAll = $<HTMLInputElement>("select-all");
const summary = $("summary");
const downloadButton = $<HTMLButtonElement>("download");
const cancelButton = $<HTMLButtonElement>("cancel");
const progress = $("progress");
const progressBar = $("progress-bar");
const status = $("status");

const selected = new Set<string>();
let datasets: Dataset[] = [];
let manifest: Manifest;
let manifestText = "";
let busy = false;

themeButton(document.getElementById("theme-toggle") as HTMLButtonElement);

const card = (d: Dataset, grid: Catalog["coverage_grid"] | undefined) => datasetCard(d, grid, selected, updateSummary, () => busy);

function selection(): Dataset[] {
  return datasets.filter((d) => selected.has(d.name));
}

function allSelected(): boolean {
  return datasets.length > 0 && selected.size === datasets.length;
}

/** Files of the zip, relative to the database directory, with their size. */
function zipFiles(): { path: string; size: number }[] {
  const files = selection().flatMap((d) => d.files);
  // The index and the manifest describe the whole database: only with all datasets
  if (allSelected() && manifest.files[DB_FILE]) files.push(DB_FILE);
  return files.map((path) => ({ path, size: manifest.files[path].size }));
}

function updateSummary() {
  const n = selected.size;
  selectAll.checked = allSelected();
  selectAll.indeterminate = n > 0 && !allSelected();
  const size = zipFiles().reduce((s, f) => s + f.size, 0);
  summary.textContent = n ? `${n} of ${datasets.length} datasets selected, ${formatSize(size)}` : `${datasets.length} datasets`;
  downloadButton.disabled = busy || n === 0;
}

function readme(): string {
  const chosen = selection();
  const full = allSelected();
  const lines = [
    "AntADatabase: radar isochrones (IRHs) of Antarctica",
    "",
    `Downloaded on ${new Date().toISOString().slice(0, 10)} from https://antadatabase.com/download/`,
    `Database version: ${manifest.db_version ?? "unknown"}`,
    "",
    "Each dataset directory holds raw_md.json (the metadata of its IRHs) and h5/, one HDF5",
    "(netCDF) file per radar line, with the depth of each IRH along the line (IRH_DEPTH, m).",
    "",
    ...(full
      ? [
          "Read it with Python (pip install anta_database):",
          "",
          "    from anta_database import Database",
          '    db = Database("AntADatabase")',
          "    db.query(...)",
          "",
          "Update it later with db.sync(), which only downloads the files that changed.",
        ]
      : [
          "This is a selection of datasets, without the index of the database (AntADatabase.db).",
          "To use them with the anta_database Python package, put the dataset directories in an",
          "AntADatabase directory and build the index once:",
          "",
          "    from anta_database import Database",
          '    db = Database("path/to/AntADatabase", index_database=True)',
          "    db.query(...)",
        ]),
    "",
    "Please cite the datasets you use:",
    "",
  ];
  for (const d of chosen) {
    const info = d.info;
    lines.push(`- ${info?.citation ?? d.name} (${d.name})`);
    for (const doi of info?.doi_dataset ?? []) lines.push(`    Dataset: ${doi}`);
    for (const doi of info?.doi_publication ?? []) lines.push(`    Publication: ${doi}`);
  }
  return lines.join("\n") + "\n";
}

let abort: AbortController | undefined;

async function download() {
  const files = zipFiles();
  const total = files.reduce((s, f) => s + f.size, 0);
  const name = allSelected() ? "AntADatabase.zip" : `AntADatabase_${selected.size}_datasets.zip`;
  const lastModified = manifest.created ? new Date(manifest.created) : new Date();
  // The whole database is in an AntADatabase directory; a selection has only the dataset
  // directories, to put in an existing database directory
  const full = allSelected();
  const root = full ? `${ROOT_DIR}/` : "";
  const items = [
    { name: `${root}README.txt`, text: readme(), lastModified: new Date() },
    ...(full ? [{ name: `${root}${MANIFEST_FILE}`, text: manifestText, lastModified }] : []),
    ...files.map((file) => ({ name: `${root}${file.path}`, url: `${DB_URL}/${file.path}`, size: file.size, lastModified })),
  ];

  abort = new AbortController();
  const signal = abort.signal;
  const showProgress = (done: number) => {
    progressBar.style.width = `${(100 * done) / Math.max(total, 1)}%`;
    status.textContent = `Downloading ${formatSize(done)} of ${formatSize(total)}…`;
  };
  try {
    const saved = await saveZip(name, items, {
      signal,
      onStart: () => {
        setBusy(true);
        showProgress(0);
      },
      onProgress: showProgress,
    });
    if (!saved) return;
    progressBar.style.width = "100%";
    status.textContent = full
      ? `Downloaded ${name} (${formatSize(total)}). Unzip it to get the ${ROOT_DIR} directory.`
      : `Downloaded ${name} (${formatSize(total)}), with the directories of the datasets.`;
  } catch (e) {
    if (signal.aborted) {
      status.textContent = "Download cancelled.";
    } else {
      console.error(e);
      status.textContent = `The download failed: ${(e as Error).message}. Please try again.`;
    }
  } finally {
    setBusy(false);
  }
}

function setBusy(value: boolean) {
  busy = value;
  cancelButton.hidden = !value;
  progress.hidden = !value && !status.textContent;
  selectAll.disabled = value;
  for (const box of list.querySelectorAll("input")) box.disabled = value;
  updateSummary();
}

async function init() {
  const [manifestResponse, catalog] = await Promise.all([
    fetchData(`${DB_URL}/${MANIFEST_FILE}`),
    fetchData(`${DATA_ROOT}/catalog/catalog.json`)
      .then((r) => (r.ok ? (r.json() as Promise<Catalog>) : undefined))
      .catch(() => undefined),
  ]);
  if (!manifestResponse.ok) throw new Error(`manifest: ${manifestResponse.status} ${manifestResponse.statusText}`);
  manifestText = await manifestResponse.text();
  manifest = JSON.parse(manifestText);
  if (!catalog) console.warn("No catalog.json: export it with python -m anta_database.web.catalog");

  // Datasets are the directories of the manifest
  const byName = new Map<string, Dataset>();
  for (const [path, { size }] of Object.entries(manifest.files)) {
    const slash = path.indexOf("/");
    if (slash < 0) continue;
    const name = path.slice(0, slash);
    const d = byName.get(name) ?? { name, files: [], size: 0 };
    d.files.push(path);
    d.size += size;
    byName.set(name, d);
  }
  const infos = new Map(catalog?.datasets.map((d) => [d.name, d]));
  datasets = [...byName.values()].map((d) => ({ ...d, info: infos.get(d.name) }));
  datasets.sort((a, b) => (a.info?.citation ?? a.name).localeCompare(b.info?.citation ?? b.name));

  list.replaceChildren(...datasets.map((d) => card(d, catalog?.coverage_grid)));
  const total = Object.values(manifest.files).reduce((sum, f) => sum + f.size, 0);
  $("total-size").textContent = `, ${formatSize(total)}`;
  updateSummary();
}

selectAll.addEventListener("change", () => {
  for (const d of datasets) (selectAll.checked ? selected.add(d.name) : selected.delete(d.name));
  for (const box of list.querySelectorAll<HTMLInputElement>(".dataset > input")) {
    box.checked = selectAll.checked;
    box.closest(".dataset")!.classList.toggle("selected", box.checked);
  }
  updateSummary();
});
downloadButton.addEventListener("click", () => {
  status.textContent = "";
  download().catch((e) => {
    console.error(e);
    status.textContent = `The download failed: ${(e as Error).message}`;
    setBusy(false);
  });
});
cancelButton.addEventListener("click", () => abort?.abort());

init().catch((e) => {
  console.error(e);
  list.replaceChildren(element("p", { textContent: `The datasets could not be loaded (${(e as Error).message}). Please try again later.` }));
});
