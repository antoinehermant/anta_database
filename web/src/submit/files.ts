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

export const COORDINATES = ["PSX", "PSY", "lon", "lat"];
/** Variables that may come with the layers, in metres. */
export const VARIABLES = ["ICE_THK", "SURF_ELEV", "BED_ELEV", "BASAL_UNIT"];
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
  const names = [...COORDINATES, ...VARIABLES, "Line_ID", "IRH_DEPTH"];
  const simple = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const found = names.find((n) => simple(n) === simple(column));
  if (found) return found;
  if (structure === "line" && /^irh[_ -]?\d+$/i.test(column)) return `IRH${column.match(/\d+/)![0]}`;
  const s = simple(column);
  if (/^(longitude|long)$/.test(s)) return "lon";
  if (/^latitude$/.test(s)) return "lat";
  if (/depth/.test(s)) return structure === "layer" ? "IRH_DEPTH" : "IRH1, IRH2, ...";
  if (/thick/.test(s)) return "ICE_THK";
  if (/flight|line|transect|segment|profile/.test(s)) return structure === "layer" ? "Line_ID" : undefined;
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

  const header = splitLine(lines[0], separator);
  info.header = header;
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
  // Flight_ID: the name of Line_ID before ground-based radar
  const lineColumn = column("Line_ID") >= 0 ? "Line_ID" : "Flight_ID";

  const duplicates = header.filter((c, i) => header.indexOf(c) !== i);
  if (duplicates.length) error(`Duplicate columns: ${[...new Set(duplicates)].join(", ")}.`);

  // Columns: recognised, or ignored with a hint
  const irhColumns = header.filter((c) => IRH_COLUMN.test(c));
  const known = new Set([...COORDINATES, ...VARIABLES, ...(structure === "layer" ? [lineColumn, "IRH_DEPTH"] : [])]);
  for (const c of header) {
    if (known.has(c) || (structure === "line" && IRH_COLUMN.test(c))) continue;
    const hint = standardName(c, structure);
    warning(`Column "${c}" is not a standard name and will be ignored${hint ? ` (should it be ${hint}?)` : ""}.`);
  }

  // Coordinates: PSX/PSY (EPSG:3031) or lon/lat
  const usePS = column("PSX") >= 0 && column("PSY") >= 0;
  const useLonLat = column("lon") >= 0 && column("lat") >= 0;
  if (!usePS && !useLonLat) error("Coordinates are missing: PSX and PSY (EPSG:3031, m), or lon and lat (WGS84, degrees).");

  // The content expected from the structure
  const variables = VARIABLES.filter((v) => column(v) >= 0);
  if (structure === "layer") {
    if (irhColumns.length) error(`Columns ${irhColumns.join(", ")} are for files per line: one file per layer has one IRH_DEPTH column.`);
    if (column(lineColumn) < 0) error("The Line_ID column is missing: it says which radar line each point belongs to.");
    if (column("IRH_DEPTH") >= 0) {
      info.kind = "irh";
      if (variables.length)
        error(`Put ${variables.join(", ")} in separate files (one file per variable, with Line_ID and coordinates), not in the layer files.`);
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
    if (!irhColumns.length) error("No layer columns: name the depth column of each layer IRH1, IRH2, ... (m below the ice surface).");
    info.irhColumns = irhColumns.map((c) => Number(c.match(IRH_COLUMN)![1]));
    if (!/^[A-Za-z0-9._-]+$/.test(stem(file.name)))
      error("The file name is the line ID: use only letters, digits, '.', '_' and '-'.");
  }
  if (info.issues.some((i) => i.level === "error")) return info;

  // Values
  const missing = new Set(MISSING);
  const [ix, iy] = usePS ? [column("PSX"), column("PSY")] : [column("lon"), column("lat")];
  const iLine = column(lineColumn);
  const depthColumns = structure === "layer" ? (info.kind === "irh" ? [column("IRH_DEPTH")] : []) : irhColumns.map(column);
  const thicknessColumns = variables.filter((v) => v !== "SURF_ELEV" && v !== "BED_ELEV").map(column);
  const numeric = [ix, iy, ...depthColumns, ...variables.map(column)];
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
  if (iLine >= 0 && info.lineIds.has("")) warning("Some rows have no Line_ID.");
  return info;
}
