/**
 * Dataset cards and zip downloads, shared by the download page and the download frame of a
 * private preview: the description of a dataset from the catalogue (anta_database.web.catalog),
 * a small map of its radar lines, and a zip built in the browser while the files are fetched
 * (stored, not compressed), written straight to disk where the browser allows it.
 */
import { downloadZip } from "client-zip";
import { type Catalog, type CatalogDataset, type Irh, loadMask } from "./sitedata";
import { onThemeChange } from "./theme";

export interface Dataset {
  name: string;
  info?: CatalogDataset;
  files: string[]; // relative to the database directory
  size: number;
}

export function formatSize(bytes: number): string {
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

export function element<K extends keyof HTMLElementTagNameMap>(tag: K, props: Record<string, unknown> = {}, ...children: (Node | string)[]): HTMLElementTagNameMap[K] {
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

/**
 * The card of a dataset, with its checkbox: selected while its name is in selected. onChange
 * after it was toggled; busy() while downloading, when clicking the card does nothing.
 */
export function datasetCard(
  d: Dataset,
  grid: Catalog["coverage_grid"] | undefined,
  selected: Set<string>,
  onChange: () => void,
  busy: () => boolean,
): HTMLElement {
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
    onChange();
  };
  box.addEventListener("change", update);
  // Clicking the card (outside links and details) also toggles it
  root.addEventListener("click", (e) => {
    if (busy() || e.target === box || (e.target as HTMLElement).closest("a, details")) return;
    box.checked = !box.checked;
    update();
  });
  root.classList.toggle("selected", box.checked);
  return root;
}

/** A file of a zip: a text, or fetched from a URL (its size known, for the progress). */
export type ZipItem = { name: string; lastModified: Date } & ({ text: string } | { url: string; size: number });

// File System Access API (Chromium): the zip is written to disk while it is built
type SaveFilePicker = (options: { suggestedName: string; types: { description: string; accept: Record<string, string[]> }[] }) => Promise<FileSystemFileHandle>;

/**
 * Build a zip of the items and save it as name. Asks where to save first, where the browser
 * can (only right after the click: call it from the click, before anything else waits); false
 * if that dialog was closed. onStart once the download starts, onProgress with the bytes
 * fetched so far. Throws when a file fails or the signal aborts.
 */
export async function saveZip(
  name: string,
  items: ZipItem[],
  options: { signal: AbortSignal; onStart: () => void; onProgress: (bytes: number) => void },
): Promise<boolean> {
  const { signal, onProgress } = options;
  const picker = (window as unknown as { showSaveFilePicker?: SaveFilePicker }).showSaveFilePicker;
  let writable: FileSystemWritableFileStream | undefined;
  if (picker) {
    try {
      const handle = await picker({ suggestedName: name, types: [{ description: "Zip archive", accept: { "application/zip": [".zip"] } }] });
      writable = await handle.createWritable();
    } catch (e) {
      if ((e as DOMException).name === "AbortError") return false; // closed the dialog
      console.warn("No file picker, building the zip in memory:", e);
    }
  }
  options.onStart();

  let done = 0;
  async function* entries() {
    for (const item of items) {
      if ("text" in item) {
        yield { name: item.name, input: item.text, lastModified: item.lastModified };
        continue;
      }
      const response = await fetch(item.url, { signal });
      if (!response.ok || !response.body) throw new Error(`${item.name}: ${response.status} ${response.statusText}`);
      // Count the bytes as they arrive
      const counted = response.body.pipeThrough(
        new TransformStream<Uint8Array, Uint8Array>({
          transform(chunk, controller) {
            done += chunk.byteLength;
            onProgress(done);
            controller.enqueue(chunk);
          },
        }),
      );
      yield { name: item.name, input: counted, size: item.size, lastModified: item.lastModified };
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
  } catch (e) {
    await writable?.abort().catch(() => {});
    throw e;
  }
  return true;
}
