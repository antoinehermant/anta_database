/**
 * Grounding lines and ice divides, exported by anta_database.web.outlines.
 *
 * Grounding lines: the RAISED reconstruction at 20, 15, 10 and 5 ka (Bentley et al.,
 * 2014) and the present grounding line (Bedmap3 mask), laid on the bed. The colour is
 * the time, the line style the confidence (measured, inferred, speculative), and the
 * RAISED scenarios (A and B differ in the Weddell Sea) can be shown one at a time.
 *
 * Ice divides: the boundaries between the IMBIE2 drainage basins, as thin walls from
 * the bed to the ice surface, with a dark line along the surface.
 *
 * The export is 2D: heights come from the terrain the viewer draws (its coarse grid,
 * interpolated like the mesh), so the lines follow the bed as it is shown.
 */
import * as THREE from "three";
import { fetchBytes, fetchData } from "./terrain";
import { LineLayer, type Hit } from "./lines";

export interface GroundingLine {
  source: "RAISED" | "Bedmap3";
  age_ka: number; // 0: present
  confidence: Confidence;
  scenarios: ("A" | "B")[];
}

export interface IceDivide {
  basins: [string, string];
  regions: [string, string];
}

type Confidence = "Measured" | "Inferred" | "Speculative";
export type Scenario = "A" | "B" | "A and B";

interface LayerFiles<F> {
  vertices: string;
  runs: string;
  features: F[];
  sources: Record<string, { citation: string; doi: string }>;
}

interface OutlinesIndex {
  grounding_lines: LayerFiles<GroundingLine>;
  divides: LayerFiles<IceDivide>;
}

/** Height (m) of the bed or ice surface at a point (km), null without data. */
export type HeightAt = (layer: "bed" | "surface", x: number, y: number) => number | null;

// Above the bed and the surface (m), so the lines are not hidden in the terrain between its vertices
const LIFT = 25;

// Colours of the times: away from the bed colours (blues, beiges) and the ages of the isochrones
export const GROUNDING_LINE_COLORS: [number, string, string][] = [
  [0, "#111418", "Present day (Bedmap3)"],
  [5, "#a23cd6", "5 ka"],
  [10, "#ffd21f", "10 ka"],
  [15, "#ff7a1a", "15 ka"],
  [20, "#e3173d", "20 ka"],
];
// Line style of each confidence: dash and gap (px), solid without
export const CONFIDENCE_STYLES: [Confidence, [number, number] | null][] = [
  ["Measured", null],
  ["Inferred", [9, 5]],
  ["Speculative", [2, 4]],
];
const DIVIDE_COLOR = "#1d232b";

let indexLoading: Promise<OutlinesIndex> | null = null;
function loadIndex(baseUrl: string): Promise<OutlinesIndex> {
  indexLoading ??= fetchData(`${baseUrl}/outlines.json`).then((response) => {
    if (!response.ok) {
      throw new Error(
        `${baseUrl}/outlines.json not found. Export it with: python -m anta_database.web.outlines --raised RAISED_PaleoIceExtents.shp --bedmap bedmap3.nc`,
      );
    }
    return response.json();
  });
  return indexLoading;
}

async function loadLines(baseUrl: string, files: LayerFiles<unknown>): Promise<{ vertices: Float32Array; runs: Uint32Array }> {
  const [vertices, runs] = await Promise.all([
    fetchBytes(`${baseUrl}/${files.vertices}`).then((b) => new Float32Array(b)),
    fetchBytes(`${baseUrl}/${files.runs}`).then((b) => new Uint32Array(b)),
  ]);
  return { vertices, runs };
}

/** 3D vertices (x, y, height of the layer + lift) of 2D vertices; null heights take the previous one. */
function lift(vertices: Float32Array, heightAt: HeightAt, layer: "bed" | "surface"): Float32Array {
  const out = new Float32Array((vertices.length / 2) * 3);
  let last = 0;
  for (let v = 0; v < vertices.length / 2; v++) {
    const [x, y] = [vertices[2 * v], vertices[2 * v + 1]];
    const h = heightAt(layer, x, y) ?? (layer === "surface" ? heightAt("bed", x, y) : null) ?? last;
    last = h;
    out.set([x, y, h + LIFT], 3 * v);
  }
  return out;
}

/** Runs of the features kept, with their feature ids unchanged. */
function keepRuns(runs: Uint32Array, keep: (feature: number) => boolean): Uint32Array {
  const out: number[] = [];
  for (let r = 0; r < runs.length; r += 3) if (keep(runs[r])) out.push(runs[r], runs[r + 1], runs[r + 2]);
  return new Uint32Array(out);
}

export class GroundingLines {
  readonly group = new THREE.Group();
  // One layer per confidence: a line style each
  readonly layers = new Map<Confidence, LineLayer>();
  features: GroundingLine[] = [];
  sources: LayerFiles<GroundingLine>["sources"] = {};
  scenario: Scenario = "A";
  // Times (ka, 0 for the present) not shown
  readonly hiddenAges = new Set<number>();
  private loading: Promise<void> | null = null;

  constructor() {
    for (const [confidence, dash] of CONFIDENCE_STYLES) {
      const layer = new LineLayer({ linewidth: 2.5, dashed: dash !== null });
      // Drawn over the bed where both are about as close
      Object.assign(layer.material, { polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4 });
      this.layers.set(confidence, layer);
      this.group.add(layer.group);
    }
  }

  get lineLayers(): LineLayer[] {
    return [...this.layers.values()];
  }

  /** Show or hide all the lines (hidden lines can't be hovered either). */
  setVisible(visible: boolean): void {
    this.group.visible = visible;
    for (const layer of this.layers.values()) layer.group.visible = visible;
  }

  /** Fetch and lay the lines on the bed, once. */
  load(baseUrl: string, heightAt: HeightAt): Promise<void> {
    this.loading ??= (async () => {
      const index = await loadIndex(baseUrl);
      const files = index.grounding_lines;
      this.features = files.features;
      this.sources = files.sources;
      const { vertices, runs } = await loadLines(baseUrl, files);
      const lines = lift(vertices, heightAt, "bed");
      const colors = new Map(GROUNDING_LINE_COLORS.map(([age, color]) => [age, new THREE.Color(color)]));
      for (const [confidence, layer] of this.layers) {
        const kept = keepRuns(runs, (f) => this.features[f].confidence === confidence);
        if (!kept.length) continue;
        const part = layer.add(confidence, lines, 3, kept);
        layer.finish();
        layer.recolor(part, (_, f, color) => color.copy(colors.get(this.features[f].age_ka) ?? colors.get(0)!));
      }
      this.applyFilter();
    })();
    return this.loading;
  }

  /** Show the lines of the scenario and of the times not hidden. */
  applyFilter(): void {
    for (const layer of this.layers.values()) {
      if (!layer.loaded) continue;
      layer.setHidden((f) => {
        const line = this.features[f];
        if (!line) return true;
        if (this.hiddenAges.has(line.age_ka)) return true;
        if (line.source === "Bedmap3") return false;
        return this.scenario !== "A and B" && !line.scenarios.includes(this.scenario);
      });
    }
  }

  /** Dashes of the same length on screen at any zoom, from the size of a pixel (km). */
  setDashPixel(pixel: number): void {
    for (const [confidence, dash] of CONFIDENCE_STYLES) if (dash) this.layers.get(confidence)!.setDashPixel(pixel, ...dash);
  }

  /** Tooltip of a hovered line: title, rows and footnote. */
  describe(layer: LineLayer, hit: Hit): { title: string; rows: [string, string][]; note: string } {
    const line = this.features[hit.feature];
    const bed = Math.round(layer.elevation(hit.segment, hit.end) - LIFT);
    if (line.source === "Bedmap3") {
      return {
        title: "Present grounding line",
        rows: [["Bed elevation", `${bed} m`]],
        note: `Edge of grounded ice on the ${this.sources.Bedmap3?.citation ?? "Bedmap3 mask"}`,
      };
    }
    const scenarios = line.scenarios.length === 2 ? "A and B" : `${line.scenarios[0]} only`;
    return {
      title: `Grounding line at ${line.age_ka} ka`,
      rows: [
        ["Confidence", line.confidence],
        ["Scenarios", scenarios],
        ["Bed elevation (today)", `${bed} m`],
      ],
      note: `${this.sources.RAISED?.citation ?? "RAISED Consortium"}: scenarios A and B differ in the Weddell Sea sector`,
    };
  }
}

export class IceDivides {
  readonly group = new THREE.Group();
  // Dark line along the ice surface, for hover
  readonly lines = new LineLayer({ linewidth: 2, color: DIVIDE_COLOR });
  features: IceDivide[] = [];
  sources: LayerFiles<IceDivide>["sources"] = {};
  private walls = new THREE.Mesh(
    new THREE.BufferGeometry(),
    new THREE.MeshBasicMaterial({ color: DIVIDE_COLOR, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false }),
  );
  // Ice thickness (m) at each vertex of the line, for the tooltip
  private thickness = new Float32Array(0);
  private loading: Promise<void> | null = null;

  constructor() {
    this.walls.frustumCulled = false;
    this.walls.renderOrder = 2; // after the transparent ice surface, which they cross
    this.lines.group.renderOrder = 3;
    this.group.add(this.walls, this.lines.group);
  }

  /** Fetch the divides and build their walls, once. */
  load(baseUrl: string, heightAt: HeightAt): Promise<void> {
    this.loading ??= (async () => {
      const index = await loadIndex(baseUrl);
      const files = index.divides;
      this.features = files.features;
      this.sources = files.sources;
      const { vertices, runs } = await loadLines(baseUrl, files);
      const top = lift(vertices, heightAt, "surface");
      const n = vertices.length / 2;
      this.thickness = new Float32Array(n);
      // Walls: a quad between consecutive vertices, from the bed to the surface
      const positions = new Float32Array(n * 6);
      for (let v = 0; v < n; v++) {
        const [x, y] = [vertices[2 * v], vertices[2 * v + 1]];
        const surface = top[3 * v + 2] - LIFT;
        const bed = Math.min(heightAt("bed", x, y) ?? surface, surface);
        this.thickness[v] = surface - bed;
        positions.set([x, y, bed, x, y, surface], 6 * v);
      }
      const indices: number[] = [];
      for (let r = 0; r < runs.length; r += 3) {
        const [first, count] = [runs[r + 1], runs[r + 2]];
        for (let v = first; v < first + count - 1; v++) {
          const [a, b, c, d] = [2 * v, 2 * v + 1, 2 * v + 2, 2 * v + 3]; // bottom, top of v and v + 1
          indices.push(a, c, b, b, c, d);
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setIndex(indices);
      this.walls.geometry.dispose();
      this.walls.geometry = geometry;
      this.lines.add("divides", top, 3, runs);
      this.lines.finish();
    })();
    return this.loading;
  }

  /** Show or hide the walls and lines (hidden lines can't be hovered either). */
  setVisible(visible: boolean): void {
    this.group.visible = visible;
    this.lines.group.visible = visible;
  }

  setExaggeration(exaggeration: number): void {
    this.walls.scale.z = exaggeration / 1000;
    this.lines.setExaggeration(exaggeration);
  }

  /** Tooltip of a hovered divide: title, rows and footnote. */
  describe(hit: Hit): { title: string; rows: [string, string][]; note: string } {
    const divide = this.features[hit.feature];
    const { vertex } = this.lines.vertexOf(hit.segment, hit.end);
    const basin = (k: 0 | 1) => `${divide.basins[k]} (${divide.regions[k]})`;
    return {
      title: "Ice divide",
      rows: [
        ["Between the basins", `${basin(0)} and ${basin(1)}`],
        ["Surface elevation", `${Math.round(this.lines.elevation(hit.segment, hit.end) - LIFT)} m`],
        ["Ice thickness", `${Math.round(this.thickness[vertex])} m`],
      ],
      note: `Boundary between two drainage basins: ${this.sources.IMBIE?.citation ?? "IMBIE2 basins"}`,
    };
  }
}
