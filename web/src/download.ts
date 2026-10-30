/**
 * Download page: choose datasets and download them in one zip file.
 *
 * The descriptions come from the catalogue exported with the viewer data
 * (python -m anta_database.web.catalog), the files from the manifest of the
 * published database, so the download always matches the bucket. The zip is
 * built in the browser while the files are fetched (stored, not compressed),
 * and written straight to disk where the browser allows it.
 */
import { downloadZip } from "client-zip";
import { type Catalog, type CatalogDataset, type Irh, DATA_ROOT, fetchData, loadMask } from "./sitedata";
import { onThemeChange, themeButton } from "./theme";

// Published database (bucket prefix of manifest.json), as in anta_database.cloud.sync
const DB_URL = (
  import.meta.env.VITE_DB_URL ||
  "https://s3.pub1.infomaniak.cloud/object/v1/AUTH_2030d6d085ec4b16a7c2320f0bb9427d/antadatabase/v1"
).replace(/\/$/, "");
const ROOT_DIR = "AntADatabase";
const DB_FILE = "AntADatabase.db";
const MANIFEST_FILE = "manifest.json";

interface Manifest {
  db_version?: string;
  created?: string;
  files: Record<string, { sha256: string; size: number }>;
}

interface Dataset {
  name: string;
  info?: CatalogDataset;
  files: string[]; // relative to the database directory
  size: number;
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

function formatSize(bytes: number): string {
  if (bytes < 1e6) return `${(bytes / 1e3).toFixed(0)} kB`;
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

function formatAge(irh: Irh): string {
  if (irh.age === null) return `IRH ${irh.irh}: no age`;
  const ka = (a: number) => (a / 1000).toLocaleString("en", { maximumFractionDigits: a < 10000 ? 2 : 1 });
  return irh.age_unc === null ? `${ka(irh.age)} ka` : `${ka(irh.age)} ± ${ka(irh.age_unc)} ka`;
}

function joinWords(items: string[]): string {
  return items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

function description(d: CatalogDataset): string {
  const ages = d.irhs.map((i) => i.age).filter((a): a is number => a !== null);
  const span = ages.length
    ? `, ${Math.min(...ages) === Math.max(...ages) ? "" : `from ${(Math.min(...ages) / 1000).toLocaleString("en", { maximumFractionDigits: 1 })} to `}${(Math.max(...ages) / 1000).toLocaleString("en", { maximumFractionDigits: 1 })} ka old,`
    : "";
  const irhs = `${d.irhs.length} isochrone${d.irhs.length > 1 ? "s" : ""}${span}`;
  const lines = `${d.n_lines} radar line${d.n_lines > 1 ? "s" : ""}`;
  const where = d.regions.length ? ` in the ${joinWords(d.regions)}` : "";
  const when = d.acquisition_years ? `, acquired in ${d.acquisition_years}` : "";
  const who = d.institutes.length ? ` by ${joinWords(d.institutes)}` : "";
  return `${irhs} traced along ${lines} (${d.n_points.toLocaleString("en")} points)${where}${when}${who}.`;
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, props: Record<string, unknown> = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
  const e: HTMLElementTagNameMap[K] = document.createElement(tag);
  Object.assign(e, props);
  e.append(...children);
  return e;
}

function links(urls: string[]): (Node | string)[] {
  return urls.flatMap((url, i) => [
    ...(i ? [", "] : []),
    url.startsWith("http") ? element("a", { href: url, target: "_blank", rel: "noopener", textContent: url.replace(/^https?:\/\/(dx\.)?doi\.org\//, "") }) : url,
  ]);
}

// Map of the coverage of each dataset over the Bedmap3 mask
let maskImage: Promise<ImageBitmap | null> | undefined;
// Maps drawn, redrawn in the colours of a new theme
const maps: Parameters<typeof drawMap>[] = [];

themeButton(document.getElementById("theme-toggle") as HTMLButtonElement);
onThemeChange(() => {
  maskImage = undefined;
  for (const args of maps) drawMap(...args);
});

async function drawMap(canvas: HTMLCanvasElement, d: CatalogDataset, grid: Catalog["coverage_grid"]) {
  const size = 150 * Math.min(devicePixelRatio, 2);
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const background = await (maskImage ??= loadMask());
  if (background) ctx.drawImage(background, 0, 0, size, size);
  // The mask covers ±3333 km, as the coverage grid
  const scale = size / grid.n;
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#2f74c0";
  const dot = Math.max(2, scale * 1.5);
  let index = 0;
  for (const gap of d.coverage) {
    index += gap;
    const i = index % grid.n;
    const j = Math.floor(index / grid.n);
    ctx.fillRect(i * scale - dot / 2, (grid.n - 1 - j) * scale - dot / 2, dot, dot);
  }
}

function card(d: Dataset, grid: Catalog["coverage_grid"] | undefined): HTMLElement {
  const box = element("input", { type: "checkbox", checked: selected.has(d.name) });
  box.setAttribute("aria-label", `Select ${d.info?.citation ?? d.name}`);
  const root = element("article", { className: "dataset" });
  const body = element("div", { className: "body" });
  const title = element("h3", {}, d.info?.citation ?? d.name, element("span", { className: "name", textContent: `${d.name} · ${formatSize(d.size)}` }));
  body.append(title);
  const info = d.info;
  if (info) {
    body.append(element("p", { className: "description", textContent: description(info) }));
    const rows: [string, (Node | string)[]][] = [
      ["Dataset", links(info.doi_dataset)],
      ["Publication", links(info.doi_publication)],
      ["Projects", [info.projects.join(", ")]],
      ["Radar", [info.radar_instruments.join(", ")]],
      ["IMBIE basins", [info.imbie_basins.join(", ")]],
      ["Variables", [info.variables.join(", ")]],
    ];
    body.append(element("dl", {}, ...rows.filter(([, v]) => v.length && v.join("") !== "").flatMap(([k, v]) => [element("dt", { textContent: k }), element("dd", {}, ...v)])));
    const ages = element("div", { className: "ages" }, ...info.irhs.map((irh) => element("span", { textContent: formatAge(irh), title: `IRH ${irh.irh}` })));
    body.append(element("details", {}, element("summary", { textContent: `Ages of the ${info.irhs.length} IRHs` }), ages));
  } else {
    body.append(element("p", { className: "description muted", textContent: `${d.files.filter((f) => f.endsWith(".h5")).length} radar lines.` }));
  }
  root.append(box, body);
  if (info && grid) {
    const canvas = element("canvas", { title: `Radar lines of ${info.citation}` });
    root.append(canvas);
    maps.push([canvas, info, grid]);
    drawMap(canvas, info, grid);
  }
  const update = () => {
    if (box.checked) selected.add(d.name);
    else selected.delete(d.name);
    root.classList.toggle("selected", box.checked);
    updateSummary();
  };
  box.addEventListener("change", update);
  // Clicking the card (outside links and details) also toggles it
  root.addEventListener("click", (e) => {
    if (busy || e.target === box || (e.target as HTMLElement).closest("a, details")) return;
    box.checked = !box.checked;
    update();
  });
  root.classList.toggle("selected", box.checked);
  return root;
}

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

// File System Access API (Chromium): the zip is written to disk while it is built
type SaveFilePicker = (options: { suggestedName: string; types: { description: string; accept: Record<string, string[]> }[] }) => Promise<FileSystemFileHandle>;

let abort: AbortController | undefined;

async function download() {
  const files = zipFiles();
  const total = files.reduce((s, f) => s + f.size, 0);
  const name = allSelected() ? "AntADatabase.zip" : `AntADatabase_${selected.size}_datasets.zip`;

  // Ask where to save first: the browser only allows it right after the click
  const picker = (window as unknown as { showSaveFilePicker?: SaveFilePicker }).showSaveFilePicker;
  let writable: FileSystemWritableFileStream | undefined;
  if (picker) {
    try {
      const handle = await picker({ suggestedName: name, types: [{ description: "Zip archive", accept: { "application/zip": [".zip"] } }] });
      writable = await handle.createWritable();
    } catch (e) {
      if ((e as DOMException).name === "AbortError") return; // closed the dialog
      console.warn("No file picker, building the zip in memory:", e);
    }
  }

  abort = new AbortController();
  const signal = abort.signal;
  setBusy(true);
  let done = 0;
  const showProgress = () => {
    progressBar.style.width = `${(100 * done) / Math.max(total, 1)}%`;
    status.textContent = `Downloading ${formatSize(done)} of ${formatSize(total)}…`;
  };
  showProgress();

  const lastModified = manifest.created ? new Date(manifest.created) : new Date();
  // The whole database is in an AntADatabase directory; a selection has only the dataset
  // directories, to put in an existing database directory
  const full = allSelected();
  const root = full ? `${ROOT_DIR}/` : "";
  async function* entries() {
    yield { name: `${root}README.txt`, input: readme(), lastModified: new Date() };
    if (full) yield { name: `${root}${MANIFEST_FILE}`, input: manifestText, lastModified };
    for (const file of files) {
      const response = await fetch(`${DB_URL}/${file.path}`, { signal });
      if (!response.ok || !response.body) throw new Error(`${file.path}: ${response.status} ${response.statusText}`);
      // Count the bytes as they arrive
      const counted = response.body.pipeThrough(
        new TransformStream<Uint8Array, Uint8Array>({
          transform(chunk, controller) {
            done += chunk.byteLength;
            showProgress();
            controller.enqueue(chunk);
          },
        }),
      );
      yield { name: `${root}${file.path}`, input: counted, size: file.size, lastModified };
    }
  }

  try {
    const zip = downloadZip(entries());
    if (writable) {
      await zip.body!.pipeTo(writable, { signal });
    } else {
      // Other browsers: the zip is built in memory, then saved
      const blob = await zip.blob();
      if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
      const url = URL.createObjectURL(blob);
      element("a", { href: url, download: name }).click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    }
    done = total;
    progressBar.style.width = "100%";
    status.textContent = full
      ? `Downloaded ${name} (${formatSize(total)}). Unzip it to get the ${ROOT_DIR} directory.`
      : `Downloaded ${name} (${formatSize(total)}), with the directories of the datasets.`;
  } catch (e) {
    if (signal.aborted) {
      status.textContent = "Download cancelled.";
      await writable?.abort().catch(() => {});
    } else {
      console.error(e);
      status.textContent = `The download failed: ${(e as Error).message}. Please try again.`;
      await writable?.abort().catch(() => {});
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
