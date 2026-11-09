/**
 * Bedmap survey lines (BEDMAP1, 2 and 3), exported by anta_database.web.bedmap: the radar data,
 * traced or not, as tracks at the ice surface, and the measured bed along them.
 *
 * Vertices are rows of (x km, y km, surface m, bed m or NaN, ice thickness m or NaN, 1 where the
 * surface is measured), packed per run with runs of (track id, first vertex, vertex count). The
 * surface tracks are coloured by dataset; the bed lines by bed elevation, in the colours of the
 * bed, lifted so they show on it. A dataset is fetched when first shown. Lines with their own
 * line ID have a profile; surveys stored as one line (no line IDs) have none.
 */
import * as THREE from "three";
import { fetchBytes, fetchData } from "./terrain";
import { LineLayer, type Hit, type Part } from "./lines";
import { BED_STOPS, Colormap } from "./colormaps";
import type { ProfileInfo } from "./profile";

export interface BedmapSurvey {
  survey: string;
  institute: string | null;
  project: string | null;
  "acquisition year": number | string | null;
  "radar instrument": string | null;
  citation: string | null;
  "DOI dataset": string | null;
  "DOI article": string | null;
}

export interface BedmapTrack {
  dataset: string;
  survey: string;
  line_id: string;
  citation: string | null;
  surface: string; // "SURF_ELEV", or "SURF_ELEV, else Bedmap3"
  profile?: ProfileInfo; // lines with their own line ID only
}

interface BedmapIndex {
  datasets: Record<string, { tracks: string; track_runs: string; n_vertices: number }>;
  surveys: Record<string, BedmapSurvey>;
  tracks: BedmapTrack[];
}

export const BEDMAP_DATASETS = ["BEDMAP3", "BEDMAP2", "BEDMAP1"];
// Purples: apart from the orange tracks of the isochrones and from the age colours
export const BEDMAP_COLORS: Record<string, string> = { BEDMAP3: "#5b3a8c", BEDMAP2: "#8a63b8", BEDMAP1: "#b9a2d6" };
const STRIDE = 6;
// Over the bed (m): the terrain is a 2 km average, below or above the measured bed
const BED_LIFT = 40;
// Over the ice surface (m), for the same reason; under the isochrone tracks (TRACK_LIFT, irh.ts)
const SURFACE_LIFT = 30;

export interface BedmapPoint {
  surface: number;
  bed: number | null;
  thickness: number | null;
  surfaceMeasured: boolean;
}

export class BedmapLines {
  // Transparent, as the isochrone tracks: drawn after the ice surface (also transparent), so it
  // is never blended over them; the bed lines are opaque, seen through the ice like the isochrones
  readonly tracks = new LineLayer({ linewidth: 1.5, transparent: true });
  readonly bed = new LineLayer({ linewidth: 2 });
  index: BedmapIndex | null = null;
  bedRange = 7000; // m, as the colours of the bed
  private loading = new Map<string, Promise<void>>();
  private indexLoading: Promise<BedmapIndex> | null = null;
  private baseUrl = "";
  private readonly bedColors = new Colormap(BED_STOPS.map(([t, c]): [number, string] => [(t + 1) / 2, c]));

  constructor() {
    this.tracks.group.renderOrder = 3;
  }

  /** The datasets in the export, newest first. */
  get datasets(): string[] {
    return BEDMAP_DATASETS.filter((d) => this.index?.datasets[d]);
  }

  loadIndex(baseUrl: string): Promise<BedmapIndex> {
    this.baseUrl = baseUrl;
    this.indexLoading ??= fetchData(`${baseUrl}/bedmap.json`).then(async (response) => {
      if (!response.ok) {
        throw new Error(`${baseUrl}/bedmap.json not found. Export it with: python -m anta_database.web.bedmap /path/to/BEDMAP-build --bedmap bedmap3.nc`);
      }
      this.index = (await response.json()) as BedmapIndex;
      return this.index;
    });
    return this.indexLoading;
  }

  /** Fetch datasets, each once; done when they are all there. */
  async load(baseUrl: string, names: string[]): Promise<void> {
    const index = await this.loadIndex(baseUrl);
    await Promise.all(
      names
        .filter((name) => index.datasets[name])
        .map((name) => {
          if (!this.loading.has(name)) {
            const files = index.datasets[name];
            const loaded = Promise.all([fetchBytes(`${this.baseUrl}/${files.tracks}`), fetchBytes(`${this.baseUrl}/${files.track_runs}`)]).then(
              ([vertices, runs]) => this.add(name, new Float32Array(vertices), new Uint32Array(runs)),
            );
            this.loading.set(name, loaded.catch((error) => {
              this.loading.delete(name);
              throw error;
            }));
          }
          return this.loading.get(name)!;
        }),
    );
  }

  private add(name: string, vertices: Float32Array, runs: Uint32Array): void {
    const color = new THREE.Color(BEDMAP_COLORS[name] ?? "#7a5aa8");
    for (let z = 2; z < vertices.length; z += STRIDE) vertices[z] += SURFACE_LIFT;
    const part = this.tracks.add(name, vertices, STRIDE, runs);
    this.tracks.finish();
    this.tracks.recolor(part, (_v, _f, target) => target.copy(color));
    // Bed lines: (x, y, bed + lift), runs cut where the bed is unknown
    const bedVertices = new Float32Array((vertices.length / STRIDE) * 3);
    const bedRuns: number[] = [];
    for (let v = 0; v < vertices.length / STRIDE; v++) {
      bedVertices.set([vertices[v * STRIDE], vertices[v * STRIDE + 1], vertices[v * STRIDE + 3] + BED_LIFT], v * 3);
    }
    for (let r = 0; r < runs.length; r += 3) {
      const [track, first, count] = [runs[r], runs[r + 1], runs[r + 2]];
      let start = -1;
      for (let v = first; v <= first + count; v++) {
        const known = v < first + count && Number.isFinite(vertices[v * STRIDE + 3]);
        if (known && start < 0) start = v;
        if (!known && start >= 0) {
          if (v - start >= 2) bedRuns.push(track, start, v - start);
          start = -1;
        }
      }
    }
    if (bedRuns.length) {
      const bedPart = this.bed.add(name, bedVertices, 3, new Uint32Array(bedRuns));
      this.bed.finish();
      this.recolorBed(bedPart);
    }
  }

  private recolorBed(part: Part): void {
    this.bed.recolor(part, (vertex, _feature, target) =>
      this.bedColors.at((part.vertices[vertex * 3 + 2] - BED_LIFT) / (2 * this.bedRange) + 0.5, target),
    );
  }

  /** Colours of the bed lines from -range to +range (m), as the bed. */
  setBedRange(range: number): void {
    this.bedRange = range;
    for (const part of this.bed.parts) this.recolorBed(part);
  }

  /** Show only the datasets not hidden. */
  applyFilter(hidden: Set<string>): void {
    const isHidden = (t: number) => !this.index?.tracks[t] || hidden.has(this.index.tracks[t].dataset);
    this.tracks.setHidden(isHidden);
    this.bed.setHidden(isHidden);
  }

  point(layer: LineLayer, hit: Hit): BedmapPoint | null {
    const { part, vertex } = layer.vertexOf(hit.segment, hit.end);
    if (layer === this.bed) {
      // The bed part has the same vertices as the tracks part of the dataset
      const tracks = this.tracks.parts.find((p) => p.name === part.name);
      if (!tracks) return null;
      return this.valuesAt(tracks.vertices, vertex);
    }
    return this.valuesAt(part.vertices, vertex);
  }

  private valuesAt(vertices: Float32Array, vertex: number): BedmapPoint {
    const v = vertices.subarray(vertex * STRIDE, vertex * STRIDE + STRIDE);
    const finite = (x: number) => (Number.isFinite(x) ? x : null);
    return { surface: v[2] - SURFACE_LIFT, bed: finite(v[3]), thickness: finite(v[4]), surfaceMeasured: v[5] >= 0.5 };
  }

  /** Tooltip of a hovered track or bed line: title, rows and footnote. */
  describe(layer: LineLayer, hit: Hit): { title: string; rows: [string, string][]; note: string | null } {
    const track = this.index!.tracks[hit.feature];
    const survey = this.index!.surveys[track.survey];
    const p = this.point(layer, hit);
    const m = (x: number | null | undefined) => (x === null || x === undefined ? "not measured" : `${Math.round(x)} m`);
    const rows: [string, string][] = [
      ["Survey", track.survey],
      ["Line ID", track.line_id],
      ["Institute", survey?.institute ?? "unknown"],
      ["Acquisition year", String(survey?.["acquisition year"] ?? "unknown")],
    ];
    if (survey?.["radar instrument"]) rows.push(["Radar", survey["radar instrument"]]);
    if (p) {
      rows.push(["Surface elevation", `${Math.round(p.surface)} m${p.surfaceMeasured ? "" : " (Bedmap3 grid)"}`]);
      rows.push(["Ice thickness", m(p.thickness)]);
      rows.push(["Bed elevation", m(p.bed)]);
    }
    return { title: `${track.dataset} survey line`, rows, note: survey?.citation ? `${survey.citation}, averaged along the line` : null };
  }
}
