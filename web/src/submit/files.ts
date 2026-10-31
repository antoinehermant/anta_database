/**
 * Reading and checking the submitted files in the browser, before anything is sent.
 *
 * The rules follow what CompileDatabase reads (anta_database/compiling/compile.py): the
 * standard column names, comma (.csv) or tab (.tab) separated files with "#" comment lines
 * (e.g. a description at the top),
 * and empty, NaN or -9999 for missing values. The checks of the package
 * (anta_database/compiling/validate.py, to come) should follow the same rules.
 */

export type Structure = "layer" | "line";
export type Level = "error" | "warning";

export interface Issue {
  level: Level;
  text: string;
}

export const COORDINATES = ["PSX", "PSY", "LON", "LAT"];
/**
 * Other names accepted for the standard ones (as ALIASES in portal/checks.py): the names of
 * the compiled data, and Flight_ID, the name of the line ID before ground-based radar.
 */
export const ALIASES: Record<string, string> = { Line_ID: "LINE_ID", Flight_ID: "LINE_ID", lon: "LON", lat: "LAT" };
/** Variables that may come with the layers, in metres. */
export const VARIABLES = ["ICE_THK", "SURF_ELEV", "BED_ELEV", "BASAL_UNIT"];
/** Of these, the reference of the depths: two of them give the third (CompileDatabase). */
export const REFERENCE = ["ICE_THK", "SURF_ELEV", "BED_ELEV"];
/**
 * Optional columns with a standard name, kept in the raw files (stored with the dataset) but
 * not compiled yet. Of a layer: with the name in a layer file, IRH<n>_<name without IRH_> in a
 * line file (IRH3_DEPTH_UNC). Text columns are not checked as numbers.
 */
export const LAYER_EXTRAS: Record<string, string> = {
  IRH_DEPTH_UNC: "depth uncertainty (m)",
  IRH_AGE: "age at each point (years)",
  IRH_AGE_UNC: "age uncertainty at each point (years)",
  IRH_TWT: "two-way travel time to the layer (µs)",
  IRH_QUALITY: "picking quality or confidence",
  IRH_POWER: "reflection power (dB)",
};
export const POINT_EXTRAS: Record<string, string> = {
  SURF_TWT: "two-way travel time to the surface (µs)",
  BED_TWT: "two-way travel time to the bed (µs)",
  TRACE_NUM: "trace number in the radargram",
  DIST: "distance along the line (m)",
  TIME: "UTC date and time of the trace (ISO 8601, e.g. 2023-12-24T10:31:05Z)",
};
const TEXT_EXTRAS = ["TIME", "IRH_QUALITY"];
/** Name of an optional column of a layer in a line file: IRH_DEPTH_UNC, 3 -> IRH3_DEPTH_UNC. */
export const lineExtra = (name: string, irh: number | string) => name.replace(/^IRH_/, `IRH${irh}_`);
const LINE_EXTRA = new RegExp(`^IRH(\\d+)_(${Object.keys(LAYER_EXTRAS).map((n) => n.slice(4)).join("|")})$`);
/** Missing values recognised by CompileDatabase. */
export const MISSING = ["", "-9999", "-9999.0", "NaN", "nan"];
export const EXTENSIONS: Record<string, string> = { csv: ",", tab: "\t" };

const IRH_COLUMN = /^IRH(\d+)$/;
/** Limits of a submission, as in the portal (portal/config.py) */
export const MAX_FILES = 100;
export const MAX_FILE_GB = 1;
export const MAX_TOTAL_GB = 5;
const GB = 1024 ** 3;
/** What to do when a limit is exceeded */
export const REDUCE =
  "Try to reduce the size of your files (remove the columns that are not needed, round the values to a sensible " +
  "precision), or contact us.";
const MAX_DEPTH = 5000; // m, deeper than any ice in Antarctica
const MAP_POINTS = 4000; // points of a file drawn on the map

export interface FileInfo {
  file: File;
  name: string;
  /** "irh": one layer (per-layer), "variable": one variable (per-layer), "line": one radar line */
  kind: "irh" | "variable" | "line" | "unknown";
  header: string[];
  preview: string[][];
  rows: number;
  /** Variable of a variable file */
  variable?: string;
  /** Variables in the columns of a layer or line file */
  variables: string[];
  /** Optional columns kept in the raw files (LAYER_EXTRAS, POINT_EXTRAS) */
  extras: string[];
  /** IRH numbers of the columns of a line file */
  irhColumns: number[];
  /** Points with a value, per IRH (line file) */
  irhPoints: Map<number, number>;
  lineIds: Set<string>;
  /** Points for the map, EPSG:3031 in km, x and y interleaved */
  points: Float32Array;
  issues: Issue[];
}

/** Line ID of a line file: its name without the extension. */
export const stem = (name: string) => name.replace(/\.[^.]*$/, "");
export const extension = (name: string) => (name.match(/\.([^.]*)$/)?.[1] ?? "").toLowerCase();

/** IRH number in the name of a layer file (IRH3.tab, LDC-VHF_IRH_3.tab, 38ka_3.csv...). */
export function irhFromName(name: string): number | undefined {
  const match = stem(name).match(/irh[_ -]?(\d+)/i) ?? stem(name).match(/(\d+)(?!.*\d)/);
  return match ? Number(match[1]) : undefined;
}

// Polar stereographic projection of EPSG:3031 (WGS84, true scale at 71°S), as pyproj
const A = 6378137.0;
const E = 0.0818191908426215;
const PHI_C = (71 * Math.PI) / 180;
const tFactor = (phi: number) =>
  Math.tan(Math.PI / 4 - phi / 2) / Math.pow((1 - E * Math.sin(phi)) / (1 + E * Math.sin(phi)), E / 2);
const SCALE = (A * Math.cos(PHI_C)) / Math.sqrt(1 - E * E * Math.sin(PHI_C) ** 2) / tFactor(PHI_C);

/** Longitude and latitude (degrees) to EPSG:3031 (m). */
export function toPolarStereo(lon: number, lat: number): [number, number] {
  const rho = SCALE * tFactor((-lat * Math.PI) / 180);
  const lambda = (lon * Math.PI) / 180;
  return [rho * Math.sin(lambda), rho * Math.cos(lambda)];
}

/** A standard name differing only in case or punctuation, for "did you mean". */
function standardName(column: string, structure: Structure): string | undefined {
  const extras = [...Object.keys(POINT_EXTRAS), ...(structure === "layer" ? Object.keys(LAYER_EXTRAS) : [])];
  const names = [...COORDINATES, ...VARIABLES, ...(structure === "layer" ? ["LINE_ID", "IRH_DEPTH"] : []), ...extras];
  const simple = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const found = names.find((n) => simple(n) === simple(column));
  if (found) return found;
  if (structure === "line" && /^irh[_ -]?\d+$/i.test(column)) return `IRH${column.match(/\d+/)![0]}`;
  const s = simple(column);
  if (/^(longitude|long)$/.test(s)) return "LON";
  if (/^latitude$/.test(s)) return "LAT";
  // Optional columns (LAYER_EXTRAS, POINT_EXTRAS), before the depth and the line
  const layer = (name: string) => (structure === "layer" ? name : lineExtra(name, "1") + ", ...");
  if (/surf/.test(s) && /twt|traveltime/.test(s)) return "SURF_TWT";
  if (/bed/.test(s) && /twt|traveltime/.test(s)) return "BED_TWT";
  if (/twt|traveltime/.test(s)) return layer("IRH_TWT");
  if (/age/.test(s) && /unc|err|sigma|std/.test(s)) return layer("IRH_AGE_UNC");
  if (/^(irh)?age$/.test(s)) return layer("IRH_AGE");
  if (/unc|err|sigma|std/.test(s) && !/thick|elev|surf|bed/.test(s)) return layer("IRH_DEPTH_UNC");
  if (/qual|conf/.test(s)) return layer("IRH_QUALITY");
  if (/power|amp|reflect/.test(s)) return layer("IRH_POWER");
  if (/trace/.test(s)) return "TRACE_NUM";
  if (/^(dist|distance)/.test(s)) return "DIST";
  if (/time|date/.test(s)) return "TIME";
  if (/depth/.test(s)) return structure === "layer" ? "IRH_DEPTH" : "IRH1, IRH2, ...";
  if (/thick/.test(s)) return "ICE_THK";
  if (/flight|line|transect|segment|profile/.test(s)) return structure === "layer" ? "LINE_ID" : undefined;
  return undefined;
}

function splitLine(line: string, separator: string): string[] {
  return line.split(separator).map((v) => v.trim().replace(/^"(.*)"$/, "$1"));
}

const plural = (n: number, word: string) => `${n.toLocaleString("en")} ${word}${n === 1 ? "" : "s"}`;

const isComment = (line: string) => line.trim() === "" || line.trimStart().startsWith("#");

/**
 * Call fn on each line of a file (and its line number, from 1), reading the file in pieces so
 * that large files fit in memory. fn returns false to stop.
 */
async function forEachLine(file: File, fn: (line: string, n: number) => boolean): Promise<void> {
  const reader = file.stream().pipeThrough(new TextDecoderStream()).getReader();
  let rest = "";
  let n = 0;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const lines = (rest + value).split("\n");
      rest = lines.pop()!;
      for (const line of lines) if (!fn(line.replace(/\r$/, ""), ++n)) return;
    }
    if (rest) fn(rest.replace(/\r$/, ""), ++n);
  } finally {
    reader.cancel().catch(() => {});
  }
}

/** Read a file and check it against the rules of the chosen structure. */
export async function readFile(file: File, structure: Structure): Promise<FileInfo> {
  const info: FileInfo = {
    file,
    name: file.name,
    kind: "unknown",
    header: [],
    preview: [],
    rows: 0,
    irhColumns: [],
    variables: [],
    extras: [],
    irhPoints: new Map(),
    lineIds: new Set(),
    points: new Float32Array(),
    issues: [],
  };
  const error = (text: string) => info.issues.push({ level: "error", text });
  const warning = (text: string) => info.issues.push({ level: "warning", text });

  if (file.size > MAX_FILE_GB * GB) {
    error(`Limit exceeded: the file is ${(file.size / GB).toFixed(1)} GB, at most ${MAX_FILE_GB} GB per file. ${REDUCE}`);
    return info;
  }
  const ext = extension(file.name);
  if (!(ext in EXTENSIONS)) {
    error(`Only .csv (comma separated) and .tab (tab separated) files are accepted, not .${ext || "(no extension)"}.`);
    return info;
  }

  // The header and the first row of data, to check the columns before reading everything
  const lines: string[] = [];
  const lineNumbers: number[] = [];
  let bytes = 0;
  await forEachLine(file, (line, n) => {
    bytes += line.length + 1;
    if (isComment(line)) return true;
    lines.push(line);
    lineNumbers.push(n);
    return lines.length < 2;
  });
  if (lines.length < 2) {
    error("The file has no data: a header line and at least one row are needed.");
    return info;
  }

  // The separator must match the extension
  const tabs = lines[0].split("\t").length - 1;
  const commas = lines[0].split(",").length - 1;
  const found = tabs > commas ? "\t" : commas > 0 ? "," : "";
  if (!found) {
    // Text above the header that is not commented out is read as the header
    if (/[\t,]/.test(lines[1]))
      error(`Line ${lineNumbers[0]} is read as the header: if it is a description, start it with "#".`);
    else error("Columns must be separated by commas (.csv) or tabs (.tab).");
    return info;
  }
  if (found !== EXTENSIONS[ext]) {
    const right = found === "\t" ? "tab separated: rename it to .tab" : "comma separated: rename it to .csv";
    error(`The extension does not match the content, the file is ${right}.`);
    return info;
  }
  const separator = found;

  info.header = splitLine(lines[0], separator);
  // The standard names, for the accepted other ones (lon -> LON); the file keeps its own
  const header = info.header.map((c) => ALIASES[c] ?? c);
  // Text above the header that is not commented out is read as the header
  const first = splitLine(lines[1], separator);
  if (first.length !== header.length) {
    error(
      `Line ${lineNumbers[0]}, read as the header, has ${header.length} columns but line ${lineNumbers[1]} has ${first.length}. ` +
        `If the file starts with a description, start each of its lines with "#".`,
    );
    return info;
  }
  const column = (name: string) => header.indexOf(name);
  const lineColumn = "LINE_ID";

  const duplicates = header.filter((c, i) => header.indexOf(c) !== i);
  if (duplicates.length)
    error(`Duplicate columns: ${[...new Set(duplicates)].join(", ")} (${info.header.filter((c, i) => duplicates.includes(header[i])).join(", ")}).`);

  // Columns: recognised, optional (kept in the raw files), or unknown (kept too, with a hint)
  const irhColumns = header.filter((c) => IRH_COLUMN.test(c));
  const known = new Set([...COORDINATES, ...VARIABLES, "acq_year", ...(structure === "layer" ? [lineColumn, "IRH_DEPTH"] : [])]);
  const isExtra = (c: string) =>
    c in POINT_EXTRAS || (structure === "layer" ? c in LAYER_EXTRAS : LINE_EXTRA.test(c));
  info.extras = header.filter(isExtra);
  const unknown: string[] = [];
  for (const c of header) {
    if (known.has(c) || isExtra(c) || (structure === "line" && IRH_COLUMN.test(c))) continue;
    const hint = standardName(c, structure);
    if (hint) warning(`Column "${c}" is not a standard name: should it be ${hint}? Otherwise it is kept in the raw files but not used.`);
    else unknown.push(c);
  }
  if (unknown.length)
    warning(
      `${unknown.length === 1 ? "Column" : "Columns"} ${unknown.map((c) => `"${c}"`).join(", ")}: not a standard name, ` +
        `kept in the raw files but not used. Please describe what ${unknown.length === 1 ? "it holds" : "they hold"} in the notes (step 6).`,
    );
  // Optional columns of a layer, without the layer
  if (structure === "line")
    for (const c of info.extras.filter((e) => LINE_EXTRA.test(e))) {
      const n = c.match(LINE_EXTRA)![1];
      if (!header.includes(`IRH${n}`)) warning(`Column ${c} without a column IRH${n}.`);
    }

  // Coordinates: PSX/PSY (EPSG:3031) or LON/LAT
  const usePS = column("PSX") >= 0 && column("PSY") >= 0;
  const useLonLat = column("LON") >= 0 && column("LAT") >= 0;
  if (!usePS && !useLonLat) error("Coordinates are missing: PSX and PSY (EPSG:3031, m), or LON and LAT (WGS84, degrees).");

  // The content expected from the structure
  const variables = VARIABLES.filter((v) => column(v) >= 0);
  if (structure === "layer") {
    if (header.some((c) => LINE_EXTRA.test(c)))
      warning("Columns like IRH3_DEPTH_UNC are for files per line: in a layer file, name them IRH_DEPTH_UNC, ....");
    if (irhColumns.length) error(`Columns ${irhColumns.join(", ")} are for files per line: one file per layer has one IRH_DEPTH column.`);
    if (column(lineColumn) < 0) error("The LINE_ID column is missing: it says which radar line each point belongs to.");
    if (column("IRH_DEPTH") >= 0) {
      info.kind = "irh";
      info.variables = variables;
    } else if (variables.length === 1) {
      info.kind = "variable";
      info.variable = variables[0];
    } else if (variables.length > 1) {
      error(`One variable per file: ${variables.join(", ")} should be in separate files.`);
    } else {
      error("No IRH_DEPTH column (depth of the layer, m below the ice surface), nor a variable (ICE_THK, SURF_ELEV, BED_ELEV or BASAL_UNIT).");
    }
  } else {
    info.kind = "line";
    info.variables = variables;
    if (!irhColumns.length) error("No layer columns: name the depth column of each layer IRH1, IRH2, ... (m below the ice surface).");
    info.irhColumns = irhColumns.map((c) => Number(c.match(IRH_COLUMN)![1]));
    if (!/^[A-Za-z0-9._-]+$/.test(stem(file.name)))
      error("The file name is the line ID: use only letters, digits, '.', '_' and '-'.");
  }
  if (info.issues.some((i) => i.level === "error")) return info;

  // Values
  const missing = new Set(MISSING);
  const [ix, iy] = usePS ? [column("PSX"), column("PSY")] : [column("LON"), column("LAT")];
  const iLine = column(lineColumn);
  const depthColumns = structure === "layer" ? (info.kind === "irh" ? [column("IRH_DEPTH")] : []) : irhColumns.map(column);
  const thicknessColumns = variables.filter((v) => v !== "SURF_ELEV" && v !== "BED_ELEV").map(column);
  const numeric = [ix, iy, ...depthColumns, ...variables.map(column),
    ...info.extras.filter((c) => !TEXT_EXTRAS.includes(c) && !/^IRH\d+_QUALITY$/.test(c)).map(column)];
  const depthOrThickness = [...depthColumns, ...thicknessColumns];
  // IRH number of the depth columns of a line file
  const irhOfColumn = new Map(structure === "line" ? depthColumns.map((c) => [c, Number(header[c].match(IRH_COLUMN)![1])]) : []);
  let notNumber = 0;
  let firstBad = "";
  let noCoordinates = 0;
  let outside = 0;
  let negative = 0;
  let tooDeep = 0;
  // About MAP_POINTS points on the map, from the size of the first lines
  const step = Math.max(1, Math.floor(file.size / (bytes / lineNumbers[1]) / MAP_POINTS));
  const points: number[] = [];
  let hashLine = 0;
  let header_ = true;

  // Comment lines start with "#" and are skipped, as blank lines. CompileDatabase reads files
  // with comment="#": a "#" anywhere else would cut the line there
  await forEachLine(file, (line, n) => {
    if (isComment(line)) return true;
    if (header_) {
      header_ = false;
      return true;
    }
    if (line.includes("#")) hashLine ||= n;
    const values = splitLine(line, separator);
    const r = ++info.rows;
    if (info.preview.length < 5) info.preview.push(values);
    for (const c of numeric) {
      const v = values[c] ?? "";
      if (!missing.has(v) && !Number.isFinite(Number(v))) {
        if (!notNumber++) firstBad = `"${v}" in ${header[c]}, line ${n}`;
      }
    }
    const a = values[ix] ?? "";
    const b = values[iy] ?? "";
    if (missing.has(a) || missing.has(b)) {
      noCoordinates++;
      return true;
    }
    let x = Number(a);
    let y = Number(b);
    if (!usePS) {
      if (y < -90 || y > -60 || x < -180 || x > 360) {
        outside++;
        return true;
      }
      [x, y] = toPolarStereo(x, y);
    } else if (Math.abs(x) > 3.4e6 || Math.abs(y) > 3.4e6) {
      outside++;
      return true;
    }
    if (r % step === 0) points.push(x / 1000, y / 1000);
    if (iLine >= 0) info.lineIds.add(values[iLine]);
    for (const c of depthOrThickness) {
      const v = values[c] ?? "";
      if (missing.has(v)) continue;
      const d = Number(v);
      if (d < 0) negative++;
      else if (d > MAX_DEPTH) tooDeep++;
      const irh = irhOfColumn.get(c);
      if (irh !== undefined) info.irhPoints.set(irh, (info.irhPoints.get(irh) ?? 0) + 1);
    }
    return true;
  });
  info.points = new Float32Array(points);
  if (structure === "line") info.lineIds.add(stem(file.name));

  if (hashLine) error(`Line ${hashLine} contains "#": only comment lines may, and they must start with it (everything after a "#" is ignored).`);
  if (notNumber) error(`${plural(notNumber, "value")} are not numbers, for example ${firstBad}. Missing values must be empty, NaN or -9999.`);
  if (outside) error(`${plural(outside, "point")} are outside Antarctica (${usePS ? "PSX, PSY must be EPSG:3031 in metres" : "latitudes must be south of 60°S"}).`);
  if (negative) error(`${plural(negative, "value")} are negative: depths and thicknesses are in metres below the ice surface (not elevations).`);
  if (tooDeep) warning(`${plural(tooDeep, "value")} are deeper than ${MAX_DEPTH} m: are they missing values? Missing values must be empty, NaN or -9999.`);
  if (noCoordinates) warning(`${plural(noCoordinates, "row")} without coordinates will be ignored.`);
  if (iLine >= 0 && info.lineIds.has("")) warning("Some rows have no LINE_ID.");
  return info;
}
