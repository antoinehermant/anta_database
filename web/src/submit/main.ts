/**
 * Submission page: a form for a dataset of radar isochrones, checked in the browser and sent
 * to the portal (portal/app.py), which writes the inputs of CompileDatabase from it.
 *
 * The fields are saved in the browser (localStorage) while typing, so a reload keeps them;
 * the files have to be added again.
 */
import { type Catalog, MAP_EXTENT_KM, loadCatalog, loadMask, DATA_ROOT, fetchData } from "../sitedata";
import { onThemeChange, themeButton } from "../theme";
import { setupVerification, token, verified } from "../verify";
import { type FileInfo, type Issue, type Structure, MAX_FILES, MAX_TOTAL_GB, REDUCE, irhFromName, readFile, stem } from "./files";

// Portal API: same origin in development (proxied by Vite to portal/app.py)
const PORTAL_URL = (import.meta.env.VITE_PORTAL_URL ?? "").replace(/\/$/, "");
// The development version of the page (/submit/?dev), linked discreetly at the bottom: open
// for tests before the portal is public (VITE_PORTAL_PUBLIC=true at the build)
const DEV_PAGE = new URLSearchParams(location.search).has("dev");
const PUBLIC = import.meta.env.VITE_PORTAL_PUBLIC === "true";
// Whether the portal takes submissions: set from its status (portal/app.py, /api/status)
let open = false;
const DRAFT_KEY = "submit-draft";

interface Draft {
  structure: Structure | null;
  publication: { first_author: string; et_al: boolean; year: string; doi_publication: string; doi_dataset: string };
  survey: { institutes: string[]; projects: string[]; radar_instruments: string[]; acquisition_year: string };
  contributor: { name: string; email: string; affiliation: string; orcid: string };
  notes: string;
  /** Redistribution: "yes", "review" (checked, kept private), "no", or "" until chosen */
  licence: "" | "yes" | "review" | "no";
  /** Age and uncertainty by IRH number */
  ages: Record<string, { age: string; age_unc: string }>;
  /** IRH number of each layer file (one file per layer) */
  fileIrh: Record<string, string>;
  /** Acquisition year and project of a file, when different from the survey */
  fileSurvey: Record<string, { acquisition_year: string; project: string }>;
}

const emptyDraft = (): Draft => ({
  structure: null,
  publication: { first_author: "", et_al: true, year: "", doi_publication: "", doi_dataset: "" },
  survey: { institutes: [], projects: [], radar_instruments: [], acquisition_year: "" },
  contributor: { name: "", email: "", affiliation: "", orcid: "" },
  notes: "",
  licence: "",
  ages: {},
  fileIrh: {},
  fileSurvey: {},
});

function loadDraft(): Draft {
  try {
    const saved = JSON.parse(localStorage.getItem(DRAFT_KEY) ?? "null");
    if (saved) {
      const draft = emptyDraft();
      for (const key of Object.keys(draft) as (keyof Draft)[]) {
        const value = saved[key];
        if (value === undefined) continue;
        const base = draft[key];
        (draft as any)[key] = base && typeof base === "object" && !Array.isArray(base) ? { ...base, ...value } : value;
      }
      // Drafts of older versions of the form
      if (!["yes", "review", "no"].includes(draft.licence)) draft.licence = "";
      // Only the structure chosen, nothing filled in: start again from the choice
      if (JSON.stringify({ ...draft, structure: null }) === JSON.stringify(emptyDraft())) draft.structure = null;
      return draft;
    }
  } catch {
    // Unreadable draft: start again
  }
  return emptyDraft();
}

let draft = loadDraft();
let files: FileInfo[] = [];
let catalog: Catalog | undefined;
let mask: Promise<ImageBitmap | null> | undefined;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const save = () => localStorage.setItem(DRAFT_KEY, JSON.stringify(draft));

function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Record<string, unknown> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  Object.assign(e, props);
  e.append(...children);
  return e;
}

// --- Fields bound to the draft ("publication.year" -> draft.publication.year)

function getPath(path: string): unknown {
  return path.split(".").reduce<any>((o, k) => o?.[k], draft);
}

function setPath(path: string, value: unknown) {
  const keys = path.split(".");
  const last = keys.pop()!;
  const parent = keys.reduce<any>((o, k) => o[k], draft);
  parent[last] = value;
}

function bindFields() {
  for (const input of document.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>("[data-bind]")) {
    const path = input.dataset.bind!;
    const checkbox = input instanceof HTMLInputElement && input.type === "checkbox";
    input.addEventListener("input", () => {
      if (input instanceof HTMLInputElement && input.type === "radio" && !input.checked) return;
      setPath(path, checkbox ? (input as HTMLInputElement).checked : input.value);
      save();
      update();
    });
  }
}

function showFields() {
  for (const input of document.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>("[data-bind]")) {
    const value = getPath(input.dataset.bind!);
    if (input instanceof HTMLInputElement && input.type === "checkbox") input.checked = Boolean(value);
    else if (input instanceof HTMLInputElement && input.type === "radio") input.checked = input.value === value;
    else input.value = String(value ?? "");
  }
  for (const box of document.querySelectorAll<HTMLElement>("[data-tags]")) showTags(box);
  for (const button of document.querySelectorAll<HTMLButtonElement>(".choice"))
    button.setAttribute("aria-pressed", String(button.dataset.structure === draft.structure));
  $("steps").hidden = !draft.structure;
}

// --- Lists of names with suggestions (institutes, projects, radar instruments)

function showTags(box: HTMLElement) {
  const values = getPath(box.dataset.tags!) as string[];
  box.querySelectorAll(".tag").forEach((t) => t.remove());
  const input = box.querySelector("input")!;
  for (const value of values) {
    const remove = element("button", { type: "button", title: `Remove ${value}`, textContent: "×" });
    remove.addEventListener("click", () => {
      setPath(box.dataset.tags!, values.filter((v) => v !== value));
      save();
      showTags(box);
      update();
    });
    box.insertBefore(element("span", { className: "tag" }, value, remove), input);
  }
}

function bindTags() {
  for (const box of document.querySelectorAll<HTMLElement>("[data-tags]")) {
    const input = box.querySelector("input")!;
    input.setAttribute("list", `suggest-${box.dataset.suggest}`);
    const add = () => {
      const value = input.value.replace(/,$/, "").trim();
      const values = getPath(box.dataset.tags!) as string[];
      if (value && !values.includes(value)) {
        values.push(value);
        save();
        showTags(box);
        update();
      }
      input.value = "";
    };
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === ",") {
        e.preventDefault();
        add();
      } else if (e.key === "Backspace" && !input.value) {
        const values = getPath(box.dataset.tags!) as string[];
        values.pop();
        save();
        showTags(box);
        update();
      }
    });
    // A suggestion picked from the list, or the field left with a name typed
    input.addEventListener("change", add);
    box.addEventListener("click", () => input.focus());
  }
}

function fillSuggestions() {
  if (!catalog) return;
  for (const key of ["institutes", "projects", "radar_instruments"] as const) {
    const names = new Set(catalog.datasets.flatMap((d) => d[key]));
    $(`suggest-${key}`).replaceChildren(
      ...[...names].sort((a, b) => a.localeCompare(b)).map((n) => element("option", { value: n })),
    );
  }
}

// --- Publication: citation and dataset name

/** "10.5194/tc-17-3461-2023" from a DOI, with or without https://doi.org/. */
function doiOf(value: string): string {
  return value.trim().replace(/^(https?:\/\/)?(dx\.)?doi\.org\//i, "").replace(/^doi:\s*/i, "");
}

const isDoi = (doi: string) => /^10\.\d{4,9}\/\S+$/.test(doi);
const doiUrl = (value: string) => (isDoi(doiOf(value)) ? `https://doi.org/${doiOf(value)}` : value.trim());

/** Family name as part of a dataset name: "Leysinger Vieli" -> "Leysinger_Vieli". */
function nameOf(author: string): string {
  return author
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .replace(/[^A-Za-z0-9 _-]/g, "")
    .trim()
    .replace(/[ -]+/g, "_");
}

interface Naming {
  dataset: string;
  citation: string;
  /** Datasets of the database with the same author and year */
  same: Catalog["datasets"];
}

function naming(): Naming | undefined {
  const { first_author, et_al, year } = draft.publication;
  const author = first_author.trim();
  if (!author || !/^\d{4}$/.test(year)) return undefined;
  const base = `${nameOf(author)}_${year}`;
  const same = (catalog?.datasets ?? []).filter((d) => d.name === base || new RegExp(`^${base}[a-z]$`).test(d.name));
  // A letter after the year when the database has the same author and year (Bodart_2025a, b...)
  let suffix = "";
  if (same.length) {
    const used = new Set(same.map((d) => d.name.slice(base.length)));
    suffix = "bcdefghijklmnopqrstuvwxyz".split("").find((l) => !used.has(l)) ?? "";
  }
  return {
    dataset: base + suffix,
    citation: `${author}${et_al ? " et al." : ""} (${year}${suffix})`,
    same,
  };
}

function publicationIssues(): Issue[] {
  const issues: Issue[] = [];
  const { first_author, year, doi_publication, doi_dataset } = draft.publication;
  if (!first_author.trim()) issues.push({ level: "error", text: "The family name of the first author is missing." });
  const now = new Date().getFullYear();
  if (!/^\d{4}$/.test(year) || Number(year) < 1950 || Number(year) > now + 1)
    issues.push({ level: "error", text: "The year of publication is missing or not a year." });
  const doi = doiOf(doi_publication);
  if (!doi && draft.licence !== "review")
    issues.push({ level: "warning", text: "No DOI of the publication: the AntADatabase holds published data, please add it once available." });
  else if (!isDoi(doi)) issues.push({ level: "error", text: `"${doi_publication}" is not a DOI (10.xxxx/...).` });
  if (!doi_dataset.trim() && draft.licence !== "review")
    issues.push({ level: "warning", text: "No DOI of the dataset: if the data is in a repository (PANGAEA, Zenodo...), please add it." });
  if (doi_dataset.trim() && !isDoi(doiOf(doi_dataset)) && !/^https?:\/\/\S+$/.test(doi_dataset.trim()))
    issues.push({ level: "error", text: "The dataset DOI is neither a DOI (10.xxxx/...) nor a URL." });

  const name = naming();
  for (const d of name?.same ?? [])
    issues.push({
      level: "warning",
      text: `The database already has ${d.citation} (${d.doi_publication.join(", ") || d.name}). If yours is another publication, it will be added as ${name!.citation}.`,
    });
  if (doi) {
    const published = (catalog?.datasets ?? []).find((d) => d.doi_publication.some((p) => doiOf(p).toLowerCase() === doi.toLowerCase()));
    if (published)
      issues.push({
        level: "warning",
        text: `This publication is already in the database as ${published.citation}. If you are sending new or corrected data, please say so in the notes.`,
      });
  }
  return issues;
}

/** Author and year from the DOI, from Crossref. */
async function fillFromDoi() {
  const status = $("doi-status");
  const doi = doiOf(draft.publication.doi_publication);
  if (!isDoi(doi)) {
    status.textContent = "Enter the DOI of the publication first (10.xxxx/...).";
    return;
  }
  status.textContent = "Looking up the DOI...";
  try {
    const response = await fetch(`https://api.crossref.org/works/${encodeURIComponent(doi)}`);
    if (!response.ok) throw new Error(response.status === 404 ? "DOI not found" : response.statusText);
    const work = (await response.json()).message;
    const authors: { family?: string; name?: string; sequence?: string }[] = work.author ?? [];
    const first = authors.find((a) => a.sequence === "first") ?? authors[0];
    const date = (work.issued ?? work["published-print"] ?? work["published-online"])?.["date-parts"]?.[0];
    if (first) draft.publication.first_author = first.family ?? first.name ?? "";
    draft.publication.et_al = authors.length > 1;
    if (date?.[0]) draft.publication.year = String(date[0]);
    save();
    showFields();
    update();
    const title = Array.isArray(work.title) ? work.title[0] : work.title;
    status.textContent = title ? `Found: ${title}` : "Found.";
  } catch (e) {
    status.textContent = `Could not look up the DOI (${(e as Error).message}): please fill in the author and year.`;
  }
}

// --- Files

function columnsHelp(structure: Structure): HTMLElement {
  const coords = "<code>PSX</code>, <code>PSY</code> (EPSG:3031, m) or <code>lon</code>, <code>lat</code> (WGS84, degrees)";
  const variables = "<code>ICE_THK</code>, <code>SURF_ELEV</code>, <code>BED_ELEV</code>, <code>BASAL_UNIT</code> (m)";
  const help = element("div", { className: "help" });
  help.innerHTML =
    structure === "layer"
      ? `<p>Name the columns with the standard names of the database (other columns are ignored):</p>
         <ul>
           <li><strong>Layer files</strong>, one per isochrone: <code>Line_ID</code>, ${coords}, and <code>IRH_DEPTH</code>.</li>
           <li><strong>Variable files</strong> (optional), one per variable: <code>Line_ID</code>, ${coords}, and one of ${variables}.
             Keep them out of the layer files, so they are not repeated for every layer.</li>
         </ul>`
      : `<p>Name the columns with the standard names of the database (other columns are ignored):</p>
         <ul>
           <li>${coords};</li>
           <li>one depth column per isochrone: <code>IRH1</code>, <code>IRH2</code>, ... (the same numbers in all files);</li>
           <li>optionally ${variables}.</li>
         </ul>
         <p>The name of each file is its line ID.</p>`;
  help.insertAdjacentHTML(
    "beforeend",
    `<p><strong>Depths are in metres below the ice surface, with the firn correction applied.</strong> Convert the
     two-way travel times first, with your wave speed in ice and your firn correction, and say which ones in the
     notes (step 6). Missing values are empty, <code>NaN</code> or <code>-9999</code>.</p>
     <p>A description may come at the top of the file, above the header, with every line starting with
     <code>#</code>: these lines are ignored. Do not use <code>#</code> anywhere else.</p>`,
  );
  const templates = element("p", {}, "Templates: ");
  const add = (name: string, content: string) => {
    const url = URL.createObjectURL(new Blob([content], { type: "text/csv" }));
    if (templates.childNodes.length > 1) templates.append(", ");
    templates.append(element("a", { href: url, download: name, textContent: name }));
  };
  if (structure === "layer") {
    add("IRH1.csv", "Line_ID,PSX,PSY,IRH_DEPTH\nF01,1352757.4,-863079.4,1008.35\n");
    add("ICE_THK.csv", "Line_ID,PSX,PSY,ICE_THK\nF01,1352757.4,-863079.4,2760.67\n");
  } else {
    add("F01.csv", "PSX,PSY,ICE_THK,IRH1,IRH2\n1352757.4,-863079.4,2760.67,1008.35,1169.2\n");
  }
  help.append(templates);
  return help;
}

async function addFiles(list: FileList | File[]) {
  if (!draft.structure) return;
  const structure = draft.structure;
  const status = $("files-status");
  status.textContent = `Reading ${list.length} file${list.length === 1 ? "" : "s"}...`;
  const added = await Promise.all([...list].map((f) => readFile(f, structure)));
  status.textContent = "";
  for (const info of added) {
    files = files.filter((f) => f.name !== info.name);
    files.push(info);
    if (info.kind === "irh" && !draft.fileIrh[info.name]) {
      const irh = irhFromName(info.name);
      if (irh !== undefined) draft.fileIrh[info.name] = String(irh);
    }
  }
  files.sort((a, b) => a.name.localeCompare(b.name, "en", { numeric: true }));
  save();
  showFiles();
  showLayers();
  update();
}

async function rereadFiles() {
  if (!draft.structure) return;
  const structure = draft.structure;
  files = await Promise.all(files.map((f) => readFile(f.file, structure)));
  showFiles();
  showLayers();
  update();
}

function issueList(issues: Issue[], where?: string): HTMLElement[] {
  return issues.map((i) =>
    element("li", { className: i.level }, ...(where ? [element("strong", {}, `${where}: `)] : []), i.text),
  );
}

function fileDescription(f: FileInfo): string {
  const lines = f.lineIds.size;
  const rows = `${f.rows.toLocaleString("en")} rows`;
  // The values are only read once the columns are right
  if (f.issues.some((i) => i.level === "error")) return rows;
  if (f.kind === "irh") return `Layer file, ${rows}, ${lines} line${lines === 1 ? "" : "s"}`;
  if (f.kind === "variable") return `${f.variable} file, ${rows}, ${lines} line${lines === 1 ? "" : "s"}`;
  if (f.kind === "line") return `Line ${stem(f.name)}, ${rows}, ${f.irhColumns.length} layers`;
  return rows;
}

function showFiles() {
  const list = $("file-list");
  list.replaceChildren(
    ...files.map((f) => {
      const remove = element("button", { type: "button", className: "remove", textContent: "Remove" });
      remove.addEventListener("click", () => {
        files = files.filter((g) => g !== f);
        showFiles();
        showLayers();
        update();
      });
      const errors = f.issues.some((i) => i.level === "error");
      const card = element(
        "article",
        { className: `file${errors ? " has-errors" : ""}` },
        element("header", {}, element("strong", {}, f.name), element("span", { className: "muted" }, fileDescription(f)), remove),
      );
      if (f.issues.length) card.append(element("ul", { className: "issues" }, ...issueList(f.issues)));
      if (f.header.length) {
        const table = element(
          "table",
          { className: "preview" },
          element("thead", {}, element("tr", {}, ...f.header.map((c) => element("th", {}, c)))),
          element("tbody", {}, ...f.preview.map((row) => element("tr", {}, ...row.map((v) => element("td", {}, v))))),
        );
        // Open when the columns need a look
        const details = element("details", { open: f.issues.length > 0 }, element("summary", {}, "First rows"));
        details.append(element("div", { className: "scroll" }, table));
        card.append(details);
      }
      return card;
    }),
  );
  // Files listed in the draft but not added again after a reload
  const reload = $("files-reload");
  const expected = Object.keys(draft.fileIrh).filter((n) => !files.some((f) => f.name === n));
  reload.hidden = files.length > 0 || expected.length === 0;
  reload.textContent = `The files have to be added again after reloading the page (${expected.slice(0, 4).join(", ")}${expected.length > 4 ? ", ..." : ""}).`;
}

// --- Layers and their ages

/** IRH numbers of the submission, in order. */
function irhNumbers(): number[] {
  const numbers = new Set<number>();
  for (const f of files) {
    if (f.kind === "irh") {
      const n = Number(draft.fileIrh[f.name]);
      if (Number.isInteger(n) && n > 0) numbers.add(n);
    } else if (f.kind === "line") f.irhColumns.forEach((n) => numbers.add(n));
  }
  return [...numbers].sort((a, b) => a - b);
}

function ageInputs(irh: number): HTMLElement[] {
  const value = (draft.ages[irh] ??= { age: "", age_unc: "" });
  return (["age", "age_unc"] as const).map((key) => {
    const input = element("input", {
      value: value[key],
      inputMode: "numeric",
      placeholder: key === "age" ? "age" : "±",
      className: `irh-${irh}-${key}`,
    });
    input.setAttribute("aria-label", `${key === "age" ? "Age" : "Age uncertainty"} of IRH ${irh} (years)`);
    input.addEventListener("input", () => {
      value[key] = input.value.trim();
      // Same layer in several files: keep their fields in step
      document.querySelectorAll<HTMLInputElement>(`.irh-${irh}-${key}`).forEach((i) => i !== input && (i.value = input.value));
      save();
      update();
    });
    return element("td", {}, input);
  });
}

function showLayers() {
  const root = $("layers");
  if (!files.some((f) => f.kind === "irh" || f.kind === "line")) {
    root.replaceChildren(element("p", { className: "muted" }, "Add your files first."));
    return;
  }
  const head = (...names: string[]) => element("thead", {}, element("tr", {}, ...names.map((n) => element("th", {}, n))));
  if (draft.structure === "layer") {
    const rows = files
      .filter((f) => f.kind === "irh")
      .map((f) => {
        const irh = element("input", { value: draft.fileIrh[f.name] ?? "", inputMode: "numeric", className: "irh-number" });
        irh.setAttribute("aria-label", `IRH number of ${f.name}`);
        // The age fields follow the IRH number: files of the same layer share them
        let ages: HTMLElement[] = [element("td"), element("td")];
        const showAges = () => {
          const n = Number(irh.value);
          const cells = Number.isInteger(n) && n > 0 ? ageInputs(n) : [element("td"), element("td")];
          ages.forEach((c, i) => c.replaceWith(cells[i]));
          ages = cells;
        };
        const own = (draft.fileSurvey[f.name] ??= { acquisition_year: "", project: "" });
        const surveyInputs = (["acquisition_year", "project"] as const).map((key) => {
          const input = element("input", {
            value: own[key],
            placeholder: key === "project" ? draft.survey.projects.join(", ") || "project" : draft.survey.acquisition_year || "year",
          });
          input.setAttribute("aria-label", `${key === "project" ? "Project" : "Acquisition year"} of ${f.name}, if different`);
          if (key === "project") input.setAttribute("list", "suggest-projects");
          input.addEventListener("input", () => {
            own[key] = input.value.trim();
            save();
            update();
          });
          return element("td", {}, input);
        });
        const row = element("tr", {}, element("td", {}, f.name), element("td", {}, irh), ...ages, ...surveyInputs);
        showAges();
        irh.addEventListener("input", () => {
          draft.fileIrh[f.name] = irh.value.trim();
          save();
          showAges();
          update();
        });
        return row;
      });
    const variables = files.filter((f) => f.kind === "variable");
    root.replaceChildren(
      element(
        "table",
        { className: "layers" },
        head("File", "IRH", "Age (years)", "± (years)", "Acquisition year*", "Project*"),
        element("tbody", {}, ...rows),
      ),
      element("p", { className: "muted" }, "* Only if different from the survey above."),
      ...(variables.length
        ? [element("p", {}, `Variables: ${variables.map((f) => `${f.variable} (${f.name})`).join(", ")}.`)]
        : []),
    );
  } else {
    const points = (irh: number) => files.reduce((n, f) => n + (f.irhPoints.get(irh) ?? 0), 0);
    const lines = (irh: number) => files.filter((f) => f.irhColumns.includes(irh)).length;
    const rows = irhNumbers().map((irh) =>
      element(
        "tr",
        {},
        element("td", {}, `IRH${irh}`),
        ...ageInputs(irh),
        element("td", { className: "muted" }, `${lines(irh)} lines, ${points(irh).toLocaleString("en")} points`),
      ),
    );
    root.replaceChildren(
      element("table", { className: "layers" }, head("Layer", "Age (years)", "± (years)", ""), element("tbody", {}, ...rows)),
    );
  }
}

function layerIssues(): Issue[] {
  const issues: Issue[] = [];
  if (!files.length) return [{ level: "error", text: "No files added." }];
  const size = files.reduce((n, f) => n + f.file.size, 0);
  if (files.length > MAX_FILES)
    issues.push({ level: "error", text: `Limit exceeded: ${files.length} files, at most ${MAX_FILES} per submission. Please contact us.` });
  if (size > MAX_TOTAL_GB * 1024 ** 3)
    issues.push({
      level: "error",
      text: `Limit exceeded: ${(size / 1024 ** 3).toFixed(1)} GB of files, at most ${MAX_TOTAL_GB} GB per submission. ${REDUCE}`,
    });
  const layers = files.filter((f) => f.kind === "irh" || f.kind === "line");
  if (!layers.length && !files.some((f) => f.issues.length)) issues.push({ level: "error", text: "No layer files." });
  if (draft.structure === "layer") {
    for (const f of files.filter((g) => g.kind === "irh")) {
      const n = Number(draft.fileIrh[f.name]);
      if (!Number.isInteger(n) || n < 1) issues.push({ level: "error", text: `${f.name}: give the IRH number of the layer (1, 2, ...).` });
      const year = draft.fileSurvey[f.name]?.acquisition_year;
      if (year && !/^\d{4}$/.test(year)) issues.push({ level: "error", text: `${f.name}: the acquisition year is not a year.` });
    }
    const variables = files.filter((f) => f.kind === "variable").map((f) => f.variable);
    const twice = variables.filter((v, i) => variables.indexOf(v) !== i);
    if (twice.length) issues.push({ level: "warning", text: `Several files for ${[...new Set(twice)].join(", ")}: fine if they cover different lines.` });
    // Variables of lines without any layer
    const withLayers = new Set(files.filter((f) => f.kind === "irh").flatMap((f) => [...f.lineIds]));
    for (const f of files.filter((g) => g.kind === "variable")) {
      const alone = [...f.lineIds].filter((id) => !withLayers.has(id));
      if (alone.length && withLayers.size)
        issues.push({ level: "warning", text: `${f.name}: ${alone.length} line(s) without any layer (${alone.slice(0, 3).join(", ")}...).` });
    }
  }
  const undated: number[] = [];
  for (const irh of irhNumbers()) {
    const { age = "", age_unc = "" } = draft.ages[irh] ?? {};
    if (!age) undated.push(irh);
    else if (!(Number(age) > 0)) issues.push({ level: "error", text: `IRH ${irh}: the age must be a positive number of years.` });
    if (age_unc && !(Number(age_unc) >= 0)) issues.push({ level: "error", text: `IRH ${irh}: the uncertainty must be a number of years.` });
    else if (age && !age_unc) issues.push({ level: "warning", text: `IRH ${irh}: no age uncertainty.` });
    if (age && Number(age) < 1000) issues.push({ level: "warning", text: `IRH ${irh}: ${age} years, are ages in years (not ka)?` });
  }
  if (undated.length) issues.push({ level: "warning", text: `Undated layers: ${undated.map((n) => `IRH ${n}`).join(", ")}.` });
  return issues;
}

const LICENCE_NO =
  "The AntADatabase only holds data it can redistribute openly, or checks it privately (review only). " +
  "Please contact us to discuss your case.";

/** The message under the choice of redistribution. */
function showLicence() {
  const message = $("licence-message");
  message.hidden = draft.licence !== "no" && draft.licence !== "review";
  if (draft.licence === "no") {
    message.replaceChildren(
      "The AntADatabase only holds data it can redistribute openly, or checks it privately (review only). ",
      element("a", { href: "/contact/?topic=submission", textContent: "Please contact us" }),
      " to discuss your case.",
    );
  } else if (draft.licence === "review") {
    message.textContent =
      "Your data will only be checked and compiled, and the result sent to you: it is not published, and it is " +
      "deleted after the review.";
  }
}

function otherIssues(): Issue[] {
  const issues: Issue[] = [];
  const { name, email, orcid } = draft.contributor;
  if (!name.trim()) issues.push({ level: "error", text: "Your name is missing." });
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.trim())) issues.push({ level: "error", text: "Your email address is missing or not valid." });
  if (orcid.trim() && !/^(https?:\/\/orcid\.org\/)?\d{4}-\d{4}-\d{4}-\d{3}[\dX]$/.test(orcid.trim()))
    issues.push({ level: "error", text: "The ORCID is not valid (0000-0000-0000-0000)." });
  if (!verified(draft.contributor.email)) issues.push({ level: "error", text: "Please verify your email address (step 6)." });
  if (!draft.licence) issues.push({ level: "error", text: "Please say whether the AntADatabase can redistribute your data." });
  if (draft.licence === "no") issues.push({ level: "error", text: LICENCE_NO });
  const { acquisition_year, institutes } = draft.survey;
  if (acquisition_year && !/^\d{4}$/.test(acquisition_year)) issues.push({ level: "error", text: "The acquisition year is not a year." });
  if (!acquisition_year) issues.push({ level: "warning", text: "No acquisition year." });
  if (!institutes.length) issues.push({ level: "warning", text: "No institute." });
  return issues;
}

// --- Review

async function drawMap() {
  const canvas = $<HTMLCanvasElement>("map");
  const size = Math.round(canvas.clientWidth * Math.min(devicePixelRatio, 2)) || 300;
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const background = await (mask ??= loadMask());
  ctx.clearRect(0, 0, size, size);
  if (background) ctx.drawImage(background, 0, 0, size, size);
  const scale = size / (2 * MAP_EXTENT_KM);
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#2f74c0";
  for (const f of files) {
    for (let i = 0; i < f.points.length; i += 2) {
      ctx.fillRect((f.points[i] + MAP_EXTENT_KM) * scale - 1, (MAP_EXTENT_KM - f.points[i + 1]) * scale - 1, 2.5, 2.5);
    }
  }
}

function summary(): string {
  const name = naming();
  const layerFiles = files.filter((f) => f.kind === "irh" || f.kind === "line");
  const lines = new Set(files.flatMap((f) => [...f.lineIds]));
  const points = layerFiles.reduce((n, f) => n + f.rows, 0);
  const parts = [
    `${files.length} file${files.length === 1 ? "" : "s"}`,
    `${lines.size} line${lines.size === 1 ? "" : "s"}`,
    `${irhNumbers().length} layers`,
    `${points.toLocaleString("en")} points`,
  ];
  return `${name ? `${name.citation}, dataset ${name.dataset}: ` : ""}${parts.join(", ")}.`;
}

function allIssues(): { where: string; issues: Issue[] }[] {
  return [
    { where: "", issues: draft.structure ? [] : [{ level: "error", text: "Choose how your files are organised." }] },
    { where: "", issues: publicationIssues() },
    ...files.map((f) => ({ where: f.name, issues: f.issues })),
    { where: "", issues: layerIssues() },
    { where: "", issues: otherIssues() },
  ];
}

let mapPending = false;

function update() {
  const name = naming();
  $("citation").textContent = name ? `Cited as ${name.citation}, dataset ${name.dataset}` : "";
  $("publication-issues").replaceChildren(...issueList(publicationIssues()));
  const groups = allIssues();
  const issues = groups.flatMap((g) => issueList(g.issues, g.where || undefined));
  const errors = groups.some((g) => g.issues.some((i) => i.level === "error"));
  $("issues").replaceChildren(
    ...(issues.length ? issues : [element("li", { className: "ok" }, "Everything looks good.")]),
  );
  $("summary").textContent = summary();
  const submit = $<HTMLButtonElement>("submit");
  submit.disabled = errors || !open;
  submit.title = !open ? "The portal is closed" : errors ? "Fix the errors first" : "";
  showVerification();
  showLicence();
  if (!mapPending) {
    mapPending = true;
    requestAnimationFrame(() => {
      mapPending = false;
      drawMap();
    });
  }
}

// --- Status of the portal

let showVerification = () => {};

/** Open or closed (portal/data/CLOSED), or closed when the portal cannot be reached. */
async function checkStatus() {
  let message = "The portal is not open for submissions yet.";
  if ((import.meta.env.DEV || PORTAL_URL) && (PUBLIC || DEV_PAGE)) {
    try {
      const response = await fetch(`${PORTAL_URL}/api/status`);
      const status = await response.json();
      open = response.ok && status.open;
      if (status.message) message = status.message;
    } catch {
      // No answer at all: a network problem, or a DNS that does not know the portal yet
      open = false;
      message = "The portal cannot be reached at the moment: please check your connection, or try again later.";
    }
  }
  $("closed-message").textContent = message;
  $("closed").hidden = open;
  for (const id of ["send-code", "check-code"]) {
    const button = $<HTMLButtonElement>(id);
    button.disabled = !open;
    button.title = open ? "" : message;
  }
  update();
}

// --- Submission

function payload() {
  const name = naming()!;
  const { publication, survey } = draft;
  return {
    structure: draft.structure,
    dataset: name.dataset,
    citation: name.citation,
    publication: {
      ...publication,
      doi_publication: publication.doi_publication.trim() ? doiUrl(publication.doi_publication) : "",
      doi_dataset: publication.doi_dataset.trim() ? doiUrl(publication.doi_dataset) : "",
    },
    survey,
    contributor: draft.contributor,
    notes: draft.notes,
    licence: draft.licence === "yes" ? "redistribute" : draft.licence === "review" ? "review" : "",
    files: files.map((f) => ({
      name: f.name,
      kind: f.kind,
      irh: f.kind === "irh" ? Number(draft.fileIrh[f.name]) : undefined,
      variable: f.variable,
      irh_columns: f.kind === "line" ? f.irhColumns : undefined,
      rows: f.rows,
      line_ids: f.lineIds.size,
      acquisition_year: f.kind === "irh" ? draft.fileSurvey[f.name]?.acquisition_year || undefined : undefined,
      project: f.kind === "irh" ? draft.fileSurvey[f.name]?.project || undefined : undefined,
    })),
    layers: irhNumbers().map((irh) => ({
      irh,
      age: draft.ages[irh]?.age ? Number(draft.ages[irh].age) : null,
      age_unc: draft.ages[irh]?.age_unc ? Number(draft.ages[irh].age_unc) : null,
    })),
    test: DEV_PAGE || import.meta.env.DEV,
    warnings: allIssues().flatMap((g) =>
      g.issues.filter((i) => i.level === "warning").map((i) => (g.where ? `${g.where}: ${i.text}` : i.text)),
    ),
  };
}

/**
 * POST a form with its upload progress (fetch cannot tell it): onProgress(sent, total) in bytes.
 * The status and the JSON answer.
 */
function upload(
  url: string,
  body: FormData,
  headers: Record<string, string>,
  onProgress: (sent: number, total: number) => void,
): Promise<{ status: number; statusText: string; result: any }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    for (const [name, value] of Object.entries(headers)) xhr.setRequestHeader(name, value);
    xhr.upload.addEventListener("progress", (e) => e.lengthComputable && onProgress(e.loaded, e.total));
    xhr.addEventListener("load", () => {
      let result = {};
      try {
        result = JSON.parse(xhr.responseText);
      } catch {
        // Not JSON, e.g. an error page of a proxy
      }
      resolve({ status: xhr.status, statusText: xhr.statusText, result });
    });
    xhr.addEventListener("error", () => reject(new Error("the connection to the portal was lost")));
    xhr.addEventListener("abort", () => reject(new Error("the sending was stopped")));
    xhr.send(body);
  });
}

const megabytes = (bytes: number) => `${(bytes / 1e6).toFixed(bytes < 1e8 ? 1 : 0)} MB`;

/** Which file is being sent, from the bytes sent so far (the files go in order, after the form). */
function sendingText(sent: number, total: number): string {
  if (sent >= total) return "All sent. The portal is now storing your files, please wait…";
  const overhead = total - files.reduce((n, f) => n + f.file.size, 0);
  let bytes = sent - overhead;
  let i = 0;
  while (i < files.length - 1 && bytes > files[i].file.size) bytes -= files[i++].file.size;
  return `Sending file ${i + 1} of ${files.length} (${files[i].name}): ${Math.floor((100 * sent) / total)} %, ` +
    `${megabytes(sent)} of ${megabytes(total)}. Please keep this page open.`;
}

/** Leaving the page during the sending would stop it. */
function warnBeforeLeaving(e: BeforeUnloadEvent) {
  e.preventDefault();
}

async function submit(event: Event) {
  event.preventDefault();
  const button = $<HTMLButtonElement>("submit");
  const status = $("submit-status");
  if (button.disabled) return;
  button.disabled = true;
  status.textContent = "Sending…";
  const progress = $<HTMLProgressElement>("submit-progress");
  const body = new FormData();
  const data = payload();
  body.append("submission", JSON.stringify(data));
  body.append("website", $<HTMLInputElement>("website").value);
  for (const f of files) body.append("files", f.file, f.name);
  progress.value = 0;
  progress.hidden = false;
  addEventListener("beforeunload", warnBeforeLeaving);
  try {
    const response = await upload(
      `${PORTAL_URL}/api/submissions`,
      body,
      { Authorization: `Bearer ${token(draft.contributor.email)}` },
      (sent, total) => {
        progress.max = total;
        progress.value = sent;
        status.textContent = sendingText(sent, total);
      },
    );
    const result = response.result;
    if (response.status < 200 || response.status >= 300)
      throw new Error(result.detail ?? `${response.status} ${response.statusText}`);
    $("receipt-id").textContent = result.id;
    $("receipt-email").textContent = draft.contributor.email;
    if (result.preview_key) {
      const link = $<HTMLAnchorElement>("receipt-preview");
      link.href = `/viewer/?preview=${encodeURIComponent(result.id)}&key=${encodeURIComponent(result.preview_key)}`;
      $("receipt-preview-text").hidden = false;
    }
    $("form").hidden = true;
    $("receipt").hidden = false;
    $("receipt").scrollIntoView();
    status.textContent = "";
  } catch (e) {
    status.textContent = `The submission failed: ${(e as Error).message}. Please try again, or contact antoine.hermant@unibe.ch.`;
    button.disabled = false;
  } finally {
    progress.hidden = true;
    removeEventListener("beforeunload", warnBeforeLeaving);
  }
}

function reset() {
  if (!confirm("Clear all the fields and files of the form?")) return;
  draft = emptyDraft();
  files = [];
  save();
  showFields();
  showFiles();
  showLayers();
  update();
}

// --- Development: fill the form from the fake submission (portal/dev/make_fake_data.py)

async function devFill() {
  const response = await fetchData(`${import.meta.env.BASE_URL}dev/fake_submission.json`);
  if (!response.ok) {
    alert("No dev/fake_submission.json: run python portal/dev/make_fake_data.py");
    return;
  }
  const fake = await response.json();
  draft.publication = { ...draft.publication, ...fake.publication };
  draft.survey = { ...draft.survey, ...fake.survey };
  draft.contributor = { ...draft.contributor, ...fake.contributor };
  draft.licence = fake.licence === true || fake.licence === "yes" ? "yes" : "";
  draft.notes = fake.notes;
  for (const l of fake.layers) draft.ages[l.irh] = { age: String(l.age ?? ""), age_unc: String(l.age_unc ?? "") };
  save();
  showFields();
  showLayers();
  update();
}

// --- Start

function chooseStructure(structure: Structure) {
  const changed = draft.structure !== structure;
  draft.structure = structure;
  save();
  showFields();
  $("columns-help").replaceChildren(columnsHelp(structure));
  if (changed) rereadFiles();
  else update();
}

function init() {
  themeButton($<HTMLButtonElement>("theme-toggle"));
  onThemeChange(() => {
    mask = undefined;
    update();
  });
  checkStatus();
  // Before the portal is public, the form is only in the development version of the page
  const form = PUBLIC || DEV_PAGE;
  $("construction").hidden = form;
  $("intro").hidden = !form;
  $("form").hidden = !form;
  $("dev-tools").hidden = !DEV_PAGE && !(import.meta.env.DEV && form);
  $("dev-banner").hidden = !DEV_PAGE;
  // The links between the two versions keep the development mode
  $<HTMLAnchorElement>("dev-link").hidden = DEV_PAGE;
  $("dev-fill").addEventListener("click", devFill);

  bindFields();
  bindTags();
  showFields();
  if (draft.structure) $("columns-help").replaceChildren(columnsHelp(draft.structure));

  for (const button of document.querySelectorAll<HTMLButtonElement>(".choice"))
    button.addEventListener("click", () => chooseStructure(button.dataset.structure as Structure));
  $("doi-fill").addEventListener("click", fillFromDoi);

  const input = $<HTMLInputElement>("file-input");
  input.addEventListener("change", () => {
    if (input.files) addFiles(input.files);
    input.value = "";
  });
  const zone = $("dropzone");
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("over");
    if (e.dataTransfer?.files) addFiles(e.dataTransfer.files);
  });

  showVerification = setupVerification(
    PORTAL_URL,
    () => draft.contributor.email,
    () => $<HTMLInputElement>("website").value,
    update,
  );
  $("form").addEventListener("submit", submit);
  $("reset").addEventListener("click", reset);
  $("new-submission").addEventListener("click", () => {
    draft = emptyDraft();
    files = [];
    save();
    location.reload();
  });

  showFiles();
  showLayers();
  update();
  loadCatalog()
    .then((c) => {
      catalog = c;
      fillSuggestions();
      update();
    })
    .catch((e) => console.warn("No catalogue:", e));
  addEventListener("resize", () => update());
}

init();
