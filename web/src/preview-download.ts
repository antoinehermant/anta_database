/**
 * Download frame of a private preview, under its progress: the compiled dataset of the
 * submission, as a card of the download page, in one zip with the index of the database, so
 * that it opens with the Python package as it is. The files are in a private bucket: the
 * portal checks the key of the preview and passes them on (/api/previews/<id>/<key>/files/).
 */
import { type Catalog, type CatalogDataset } from "./sitedata";
import { type Dataset, datasetCard, element, formatSize, saveZip } from "./datasets";
import type { PreviewStatus } from "./preview";

const ROOT_DIR = "AntADatabase";

function readme(status: PreviewStatus, info: CatalogDataset | undefined): string {
  return [
    `AntADatabase: ${status.citation} (${status.dataset}), compiled from your submission`,
    "",
    `Downloaded on ${new Date().toISOString().slice(0, 10)} from its private preview on https://antadatabase.com.`,
    "This is the dataset as it was compiled for the review of the submission.",
    "",
    `${status.dataset}/raw_md.json holds the metadata of its IRHs, and ${status.dataset}/h5/ one HDF5`,
    "(netCDF) file per radar line, with the depth of each IRH along the line (IRH_DEPTH, m).",
    `AntADatabase.db is the index of this ${ROOT_DIR} directory, for the Python package`,
    "(pip install anta_database):",
    "",
    "    from anta_database import Database",
    `    db = Database("${ROOT_DIR}")`,
    `    db.query(dataset="${status.dataset}")`,
    `    db.plot.dataset("${status.dataset}")`,
    "",
    ...(info?.irhs.length ? [`${info.irhs.length} IRHs, ${info.n_lines} radar lines, ${info.n_points.toLocaleString("en")} points.`] : []),
  ].join("\n") + "\n";
}

/** Show the frame once the preview is ready, if the compiled dataset can be downloaded. */
export async function showDownload(panel: HTMLElement, baseUrl: string, status: PreviewStatus): Promise<void> {
  const downloads = status.downloads ?? {};
  const paths = Object.keys(downloads);
  if (!paths.some((p) => p.endsWith(".h5"))) return; // compiled before the frame existed
  const catalog: Catalog | undefined = await fetch(`${baseUrl}/catalog.json`, { cache: "no-cache" })
    .then((r) => (r.ok ? r.json() : undefined))
    .catch(() => undefined);
  const info = catalog?.datasets.find((d) => d.name === status.dataset);
  const dataset: Dataset = {
    name: status.dataset,
    info,
    files: paths,
    size: Object.values(downloads).reduce((s, n) => s + n, 0),
  };

  const selected = new Set([dataset.name]);
  let busy = false;
  let abort: AbortController | undefined;
  const summary = element("span", { className: "dl-summary" });
  const downloadButton = element("button", { type: "button", className: "dl-download", textContent: "Download zip" });
  const cancelButton = element("button", { type: "button", textContent: "Cancel", hidden: true });
  const progressBar = element("div", { className: "dl-progress-bar" });
  const progress = element("div", { className: "dl-progress", hidden: true }, progressBar);
  const statusText = element("p", { className: "dl-status", role: "status" });

  const update = () => {
    summary.textContent = selected.size ? `${paths.filter((p) => p.endsWith(".h5")).length} files, ${formatSize(dataset.size)}` : "Nothing selected";
    downloadButton.disabled = busy || !selected.size;
  };
  const setBusy = (value: boolean) => {
    busy = value;
    cancelButton.hidden = !value;
    progress.hidden = !value && !statusText.textContent;
    update();
  };
  const card = datasetCard(dataset, catalog?.coverage_grid, selected, update, () => busy);

  downloadButton.addEventListener("click", async () => {
    statusText.textContent = "";
    const name = `${status.dataset}.zip`;
    const now = new Date();
    const files = Object.entries(downloads).map(([path, size]) => ({
      name: `${ROOT_DIR}/${path}`,
      url: `${baseUrl}/files/${path.split("/").map(encodeURIComponent).join("/")}`,
      size,
      lastModified: now,
    }));
    abort = new AbortController();
    const signal = abort.signal;
    const showProgress = (done: number) => {
      progressBar.style.width = `${(100 * done) / Math.max(dataset.size, 1)}%`;
      statusText.textContent = `Downloading ${formatSize(done)} of ${formatSize(dataset.size)}…`;
    };
    try {
      const saved = await saveZip(name, [{ name: `${ROOT_DIR}/README.txt`, text: readme(status, info), lastModified: now }, ...files], {
        signal,
        onStart: () => {
          setBusy(true);
          showProgress(0);
        },
        onProgress: showProgress,
      });
      if (!saved) return;
      progressBar.style.width = "100%";
      statusText.textContent = `Downloaded ${name} (${formatSize(dataset.size)}). Unzip it to get the ${ROOT_DIR} directory.`;
    } catch (e) {
      if (signal.aborted) statusText.textContent = "Download cancelled.";
      else {
        console.error(e);
        statusText.textContent = `The download failed: ${(e as Error).message}. Please try again.`;
      }
    } finally {
      setBusy(false);
    }
  });
  cancelButton.addEventListener("click", () => abort?.abort());

  const close = element("button", { type: "button", className: "close", title: "Hide", textContent: "×" });
  close.addEventListener("click", () => (panel.hidden = true));
  panel.replaceChildren(
    close,
    element("h2", { textContent: "Download your compiled dataset" }),
    element("p", {
      textContent:
        `The HDF5 files of your dataset as ${status.licence === "redistribute" ? "they will be in the database" : "compiled for the review"}, ` +
        "with the index to open them with the Python package. Available until the review is done.",
    }),
    element("section", { className: "dl-toolbar" }, summary, element("span", { className: "spacer" }), downloadButton, cancelButton, progress, statusText),
    card,
  );
  update();
  panel.hidden = false;
}
