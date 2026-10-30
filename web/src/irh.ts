/**
 * Isochrones (IRHs) and their radar tracks at the ice surface, exported by
 * anta_database.web.irh.
 *
 * Isochrones are coloured by age, depth, fractional depth or elevation. Only
 * the age colours are ready at start: elevations are the vertices themselves,
 * and depths and fractional depths are fetched (one small file per dataset)
 * the first time they are needed.
 *
 * The basal unit (the top of the echo-free zone, a layer without age) is drawn
 * with the isochrones, in one colour whatever the colouring.
 *
 * Isochrones can be filtered by dataset and age (and the basal unit), and tracks by dataset.
 */
import * as THREE from "three";
import { fetchBytes, fetchData } from "./terrain";
import { LineLayer } from "./lines";
import { AGE, AGE_LEGEND, AGE_LEGEND_SPLIT_T, AGE_SPLIT_KA, AGE_SPLIT_T, DEPTH, ELEVATION, scaleT, ticks, type Colormap, type Scale, type Tick } from "./colormaps";

export interface Feature {
  dataset: string;
  line_id: string;
  irh: number | null; // null for the basal unit
  layer?: "basal unit"; // missing for the isochrones
  age_ka: number | null;
  age_unc_ka: number | null;
  citation: string | null;
  doi_dataset: string | null;
  doi_publication: string | null;
  institute: string | null;
  surface: string; // where the surface elevation comes from: "SURF_ELEV" or "Bedmap3"
  thickness?: string; // where the ice thickness comes from: "ICE_THK" (completed with Bedmap3) or "Bedmap3"
}

export interface Track {
  dataset: string;
  line_id: string;
  citation: string | null;
  surface: string;
}

interface DatasetFiles {
  vertices: string;
  depths: string;
  fdepths?: string; // missing in exports before fractional depths
  runs: string;
  tracks: string;
  track_runs: string;
}

export interface IrhIndex {
  datasets: Record<string, DatasetFiles>;
  features: Feature[];
  tracks: Track[];
}

export async function loadIndex(baseUrl: string): Promise<IrhIndex> {
  const response = await fetchData(`${baseUrl}/irh.json`);
  if (!response.ok) {
    throw new Error(
      `${baseUrl}/irh.json not found. Export it with: python -m anta_database.web.irh /path/to/AntADatabase --bedmap bedmap3.nc`,
    );
  }
  return response.json();
}

export type ColorBy = "Age" | "Depth" | "Fractional depth" | "Elevation";
export const COLOR_BY: ColorBy[] = ["Age", "Depth", "Fractional depth", "Elevation"];

// Fractional depths are stored as uint16 fraction × FDEPTH_SCALE (see anta_database.web.irh)
const FDEPTH_SCALE = 10000;
const FDEPTH_NODATA = 65535;
const FDEPTH_SCALE_01: Scale = { min: 0, max: 1, log: false };

export interface ColorScale {
  title: string;
  colormap: Colormap;
  ticks: Tick[];
}

const NO_AGE = new THREE.Color("#b8b8b8");
// Grey-brown, darker than the bed colours and away from the age colours
export const BASAL_UNIT_COLOR = "#6e5a48";
const BASAL_UNIT = new THREE.Color(BASAL_UNIT_COLOR);

export const isBasalUnit = (f: Feature): boolean => f.layer === "basal unit";

/** Key of a feature in the age filter: its age, or "basal unit". */
export type AgeKey = number | null | "basal unit";
export const ageKey = (f: Feature): AgeKey => (isBasalUnit(f) ? "basal unit" : f.age_ka);

/** Value at a quantile of a sample of every step-th value. */
function quantile(values: ArrayLike<number>, q: number, step = 7): number {
  const sample: number[] = [];
  for (let i = 0; i < values.length; i += step) if (Number.isFinite(values[i])) sample.push(values[i]);
  sample.sort((a, b) => a - b);
  return sample[Math.min(sample.length - 1, Math.floor(q * sample.length))] ?? 0;
}

export class Isochrones {
  readonly lines = new LineLayer({ linewidth: 2.5 });
  features: Feature[] = [];
  colorBy: ColorBy = "Age";
  private index: IrhIndex | null = null;
  private baseUrl = "";
  private ageScale: Scale = { min: 1, max: 1, log: true };
  private elevationScale: Scale = { min: 0, max: 1, log: false };
  private depthScale: Scale | null = null;
  private depths = new Map<string, Uint16Array>();
  private depthsLoading: Promise<void> | null = null;
  private fdepths = new Map<string, Uint16Array>();
  private fdepthsLoading: Promise<void> | null = null;
  // Filters
  readonly hiddenDatasets = new Set<string>();
  readonly hiddenAges = new Set<AgeKey>();

  get group(): THREE.Group {
    return this.lines.group;
  }

  async load(baseUrl: string, index: IrhIndex): Promise<void> {
    this.baseUrl = baseUrl;
    this.index = index;
    this.features = index.features;
    // Log scale from the youngest to the oldest age, so close young ages differ,
    // with young IRHs over most of the colours
    const ages = this.features.filter((f) => !isBasalUnit(f)).map((f) => f.age_ka).filter((a): a is number => a !== null && a > 0);
    this.ageScale = { min: Math.min(...ages), max: Math.max(...ages), log: true, split: AGE_SPLIT_KA, splitT: AGE_SPLIT_T };

    await Promise.all(
      Object.entries(index.datasets).map(async ([name, files]) => {
        const [vertices, runs] = await Promise.all([
          fetchBytes(`${baseUrl}/${files.vertices}`).then((b) => new Float32Array(b)),
          fetchBytes(`${baseUrl}/${files.runs}`).then((b) => new Uint32Array(b)),
        ]);
        this.lines.add(name, vertices, 3, runs);
      }),
    );
    this.lines.finish();
    const elevations = new Float32Array(this.lines.parts.reduce((n, p) => n + p.vertices.length / 3, 0));
    let k = 0;
    for (const p of this.lines.parts) for (let i = 2; i < p.vertices.length; i += 3) elevations[k++] = p.vertices[i];
    const round = (v: number, up: boolean) => (up ? Math.ceil(v / 100) : Math.floor(v / 100)) * 100;
    this.elevationScale = {
      min: round(quantile(elevations, 0.01), false),
      max: round(quantile(elevations, 0.99), true),
      log: false,
    };
    this.recolor();
  }

  /** Depths of every dataset, fetched once. */
  private loadDepths(): Promise<void> {
    this.depthsLoading ??= Promise.all(
      Object.entries(this.index!.datasets).map(async ([name, files]) => {
        const depths = new Uint16Array(await fetchBytes(`${this.baseUrl}/${files.depths}`));
        this.depths.set(name, depths);
      }),
    ).then(() => {
      const all = new Float32Array(Array.from(this.depths.values()).reduce((n, d) => n + d.length, 0));
      let offset = 0;
      for (const d of this.depths.values()) {
        all.set(d, offset);
        offset += d.length;
      }
      this.depthScale = { min: 0, max: Math.ceil(quantile(all, 0.99) / 100) * 100, log: false };
    });
    return this.depthsLoading;
  }

  /** Fractional depths of every dataset, fetched once. */
  private loadFdepths(): Promise<void> {
    this.fdepthsLoading ??= Promise.all(
      Object.entries(this.index!.datasets).map(async ([name, files]) => {
        if (!files.fdepths) {
          throw new Error("No fractional depths in this export: run anta_database.web.irh again");
        }
        this.fdepths.set(name, new Uint16Array(await fetchBytes(`${this.baseUrl}/${files.fdepths}`)));
      }),
    ).then(() => undefined);
    return this.fdepthsLoading;
  }

  async setColorBy(mode: ColorBy): Promise<void> {
    this.colorBy = mode;
    if (mode === "Depth") await this.loadDepths();
    if (mode === "Fractional depth") await this.loadFdepths();
    if (this.colorBy === mode) this.recolor();
  }

  private recolor(): void {
    const mode = this.colorBy;
    for (const part of this.lines.parts) {
      const v = part.vertices;
      const depths = this.depths.get(part.name);
      const fdepths = this.fdepths.get(part.name);
      this.lines.recolor(part, (vertex, feature, color) => {
        if (isBasalUnit(this.features[feature])) return color.copy(BASAL_UNIT);
        if (mode === "Fractional depth" && fdepths) {
          const f = fdepths[vertex];
          return f === FDEPTH_NODATA ? color.copy(NO_AGE) : DEPTH.at(f / FDEPTH_SCALE, color);
        }
        if (mode === "Elevation") return ELEVATION.at(scaleT(this.elevationScale, v[vertex * 3 + 2]), color);
        if (mode === "Depth" && depths && this.depthScale) return DEPTH.at(scaleT(this.depthScale, depths[vertex]), color);
        const age = this.features[feature].age_ka;
        return age === null ? color.copy(NO_AGE) : AGE.at(scaleT(this.ageScale, age), color);
      });
    }
  }

  /** Title, colormap and ticks of the current colours. */
  colorScale(): ColorScale {
    if (this.colorBy === "Elevation") {
      return { title: "Isochrone elevation (m)", colormap: ELEVATION, ticks: ticks(this.elevationScale) };
    }
    if (this.colorBy === "Fractional depth") {
      return { title: "Isochrone fractional depth (of ice thickness)", colormap: DEPTH, ticks: ticks(FDEPTH_SCALE_01) };
    }
    if (this.colorBy === "Depth" && this.depthScale) {
      return { title: "Isochrone depth (m)", colormap: DEPTH, ticks: ticks(this.depthScale) };
    }
    // Same colours, with the old part wider than on the lines
    const legendScale = { ...this.ageScale, splitT: AGE_LEGEND_SPLIT_T };
    return { title: "Isochrone age (ka)", colormap: AGE_LEGEND, ticks: ticks(legendScale) };
  }

  /** Depth (m) of a segment end, null until depths are loaded. */
  depth(segment: number, end: 0 | 1): number | null {
    const { part, vertex } = this.lines.vertexOf(segment, end);
    return this.depths.get(part.name)?.[vertex] ?? null;
  }

  /** Fractional depth of a segment end, null until loaded or where unknown. */
  fractionalDepth(segment: number, end: 0 | 1): number | null {
    const { part, vertex } = this.lines.vertexOf(segment, end);
    const f = this.fdepths.get(part.name)?.[vertex];
    return f === undefined || f === FDEPTH_NODATA ? null : f / FDEPTH_SCALE;
  }

  /** Whether a basal unit is shown (not filtered out). */
  showsBasalUnit(): boolean {
    return this.features.some((f) => isBasalUnit(f) && !this.hiddenDatasets.has(f.dataset) && !this.hiddenAges.has("basal unit"));
  }

  /** Show only the isochrones of the datasets and ages not hidden. */
  applyFilter(): void {
    this.lines.setHidden((f) => {
      const feature = this.features[f];
      return !feature || this.hiddenDatasets.has(feature.dataset) || this.hiddenAges.has(ageKey(feature));
    });
  }
}

/** Values of a track at a point. */
export interface TrackPoint {
  surface: number; // m
  irhNum: number;
}

/** Radar tracks at the ice surface, where IRHs are traced. */
export class Tracks {
  readonly lines = new LineLayer({ linewidth: 2, color: "#8b929b", transparent: true, dashed: true });
  tracks: Track[] = [];
  // Ages (ka) of the IRHs of each dataset, youngest first, number of IRHs without age, and
  // whether it has a basal unit (not an IRH)
  readonly datasetAges = new Map<string, { ages: number[]; noAge: number; basalUnit: boolean }>();
  private loading: Promise<void> | null = null;

  get group(): THREE.Group {
    return this.lines.group;
  }

  /** Fetch the tracks of every dataset, once. */
  load(baseUrl: string, index: IrhIndex): Promise<void> {
    this.loading ??= (async () => {
      this.tracks = index.tracks;
      const irhs = new Map<string, Map<number, number | null>>();
      const basalUnit = new Set<string>();
      for (const f of index.features) {
        if (isBasalUnit(f)) {
          basalUnit.add(f.dataset);
          continue;
        }
        if (!irhs.has(f.dataset)) irhs.set(f.dataset, new Map());
        irhs.get(f.dataset)!.set(f.irh!, f.age_ka);
      }
      for (const [dataset, ages] of irhs) {
        const known = [...new Set([...ages.values()].filter((a): a is number => a !== null))].sort((a, b) => a - b);
        this.datasetAges.set(dataset, {
          ages: known,
          noAge: [...ages.values()].filter((a) => a === null).length,
          basalUnit: basalUnit.has(dataset),
        });
      }
      await Promise.all(
        Object.entries(index.datasets).map(async ([name, files]) => {
          const [vertices, runs] = await Promise.all([
            fetchBytes(`${baseUrl}/${files.tracks}`).then((b) => new Float32Array(b)),
            fetchBytes(`${baseUrl}/${files.track_runs}`).then((b) => new Uint32Array(b)),
          ]);
          this.lines.add(name, vertices, 5, runs);
        }),
      );
      this.lines.finish();
    })();
    return this.loading;
  }

  /** Show only the tracks of the datasets not hidden. */
  applyFilter(hiddenDatasets: Set<string>): void {
    this.lines.setHidden((t) => !this.tracks[t] || hiddenDatasets.has(this.tracks[t].dataset));
  }

  point(segment: number, end: 0 | 1): TrackPoint {
    const { part, vertex } = this.lines.vertexOf(segment, end);
    const v = part.vertices.subarray(vertex * 5, vertex * 5 + 5);
    return { surface: v[2], irhNum: v[3] };
  }
}
