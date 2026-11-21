/**
 * The outlines of the regions chosen in the filter: a place as the circle of its radius (the lines
 * within it are at the place), an ice sheet or basin by its outline (regions_outlines.json.gz of
 * the export, anta_database.web.regions).
 *
 * Drawn like the ice divides (outlines.ts): a black dashed line on the ice surface, and a thin
 * transparent wall from it down to the bed through the ice. One feature per region in the
 * lines, added the first time it is chosen, the others hidden; the walls are those of the regions
 * shown. Hovered like the other lines (main.ts). Heights come from the terrain the viewer draws; over the sea, the surface at sea level.
 */
import * as THREE from "three";
import { LineLayer } from "./lines";
import type { HeightAt } from "./outlines";
import type { Region } from "./irh";

const COLOR = "#000000";
const STEP = 2; // km, at most between two vertices, so the lines follow the terrain
const LIFT = 25; // m above the surface, so the line is not hidden in the terrain between its vertices
const DASH: [number, number] = [8, 6]; // px

/** Rings (x, y km) of the outline of a region; a place: the circle of its radius. */
function rings(region: Region, outlines: Record<string, number[][]>): number[][] {
  if (region.kind !== "place") return region.key ? outlines[region.key] ?? [] : [];
  const [xmin, xmax, ymin, ymax] = region.box.map((v) => v / 1000);
  const [cx, cy, r] = [(xmin + xmax) / 2, (ymin + ymax) / 2, region.radius_km ?? (xmax - xmin) / 2];
  const n = Math.max(64, Math.ceil((2 * Math.PI * r) / STEP));
  const ring: number[] = [];
  for (let i = 0; i <= n; i++) ring.push(cx + r * Math.cos((2 * Math.PI * i) / n), cy + r * Math.sin((2 * Math.PI * i) / n));
  return [ring];
}

/** The outline of a region at the surface (x, y km, height m + LIFT), the bed under it (m), and its runs. */
interface Outline {
  top: Float32Array;
  bottom: Float32Array;
  runs: Uint32Array; // feature, first vertex, vertex count
}

function build(region: Region, feature: number, heightAt: HeightAt, outlines: Record<string, number[][]>): Outline {
  const top: number[] = [];
  const bottom: number[] = [];
  const runs: number[] = [];
  for (const ring of rings(region, outlines)) {
    const first = top.length / 3;
    for (let i = 0; i < ring.length; i += 2) {
      // Points every STEP km at most along the segment from the previous vertex
      const [x, y] = [ring[i], ring[i + 1]];
      const n = i ? Math.ceil(Math.hypot(x - ring[i - 2], y - ring[i - 1]) / STEP) : 1;
      for (let k = 1; k <= n; k++) {
        const px = i ? ring[i - 2] + ((x - ring[i - 2]) * k) / n : x;
        const py = i ? ring[i - 1] + ((y - ring[i - 1]) * k) / n : y;
        const bed = heightAt("bed", px, py);
        const surface = heightAt("surface", px, py) ?? Math.max(bed ?? 0, 0);
        top.push(px, py, surface + LIFT);
        bottom.push(Math.min(bed ?? surface, surface));
      }
    }
    runs.push(feature, first, top.length / 3 - first);
  }
  return { top: new Float32Array(top), bottom: new Float32Array(bottom), runs: new Uint32Array(runs) };
}

export class RegionOutlines {
  readonly group = new THREE.Group();
  readonly surface = new LineLayer({ linewidth: 2, color: COLOR, dashed: true });
  private walls = new THREE.Mesh(
    new THREE.BufferGeometry(),
    new THREE.MeshBasicMaterial({ color: COLOR, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthWrite: false }),
  );
  private outlines = new Map<Region, Outline>();
  // The region of each feature
  private regions: Region[] = [];

  constructor() {
    this.walls.frustumCulled = false;
    this.walls.renderOrder = 2; // after the transparent ice surface, which they cross
    this.surface.group.renderOrder = 3;
    this.group.add(this.walls, this.surface.group);
  }

  get lineLayers(): LineLayer[] {
    return [this.surface];
  }

  /** Show the outlines of these regions only. */
  show(regions: Region[], heightAt: HeightAt, outlines: Record<string, number[][]>): void {
    let added = false;
    for (const region of regions) {
      if (this.outlines.has(region)) continue;
      const outline = build(region, this.outlines.size, heightAt, outlines);
      this.outlines.set(region, outline);
      this.regions.push(region);
      if (!outline.runs.length) continue;
      this.surface.add(region.name, outline.top, 3, outline.runs);
      added = true;
    }
    if (added) for (const layer of this.lineLayers) layer.finish();
    const shown = new Set(regions.map((r) => this.outlines.get(r)!.runs[0]));
    for (const layer of this.lineLayers) {
      if (layer.loaded) layer.setHidden((f) => !shown.has(f));
      layer.invalidate();
    }
    this.buildWalls(regions.map((r) => this.outlines.get(r)!));
  }

  /** Walls of these outlines: a quad between consecutive vertices, from the bed to the surface. */
  private buildWalls(shown: Outline[]): void {
    const positions: number[] = [];
    const indices: number[] = [];
    for (const { top, bottom, runs } of shown) {
      for (let r = 0; r < runs.length; r += 3) {
        const [first, count] = [runs[r + 1], runs[r + 2]];
        const base = positions.length / 3;
        for (let v = first; v < first + count; v++) {
          positions.push(top[3 * v], top[3 * v + 1], bottom[v], top[3 * v], top[3 * v + 1], top[3 * v + 2] - LIFT);
        }
        for (let k = 0; k < count - 1; k++) {
          const [a, b, c, d] = [base + 2 * k, base + 2 * k + 1, base + 2 * k + 2, base + 2 * k + 3]; // bottom, top of k and k + 1
          indices.push(a, c, b, b, c, d);
        }
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
    geometry.setIndex(indices);
    this.walls.geometry.dispose();
    this.walls.geometry = geometry;
  }

  /** The region of a hovered feature. */
  region(feature: number): Region {
    return this.regions[feature];
  }

  /** Surface elevation (m) of a hovered segment end. */
  height(layer: LineLayer, segment: number, end: 0 | 1): number {
    return layer.elevation(segment, end) - LIFT;
  }

  setExaggeration(exaggeration: number): void {
    this.walls.scale.z = exaggeration / 1000;
    for (const layer of this.lineLayers) layer.setExaggeration(exaggeration);
  }

  /** Dashes of the same length on screen at any zoom, from the size of a pixel (km). */
  setDashPixel(pixel: number): void {
    for (const layer of this.lineLayers) layer.setDashPixel(pixel, ...DASH);
  }
}
