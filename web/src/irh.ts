/**
 * Isochrones (IRHs) exported by anta_database.web.irh, drawn as 3D lines.
 *
 * Each feature is one (flight line, IRH). Vertices are (x km, y km, elevation m);
 * the group is scaled in z by exaggeration / 1000, like the terrain.
 *
 * Hover works in screen space: once the view is still, all segments are
 * projected to the screen and binned in a grid of small cells. The hovered
 * line is the closest one within a few pixels of the cursor, the nearest to
 * the camera when several are about as close.
 */
import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { fetchBytes } from "./terrain";

export interface Feature {
  dataset: string;
  flight_id: string;
  irh: number;
  age_ka: number | null;
  age_unc_ka: number | null;
  citation: string | null;
  doi_dataset: string | null;
  doi_publication: string | null;
  institute: string | null;
  surface: string; // where the surface elevation comes from: "SURF_ELEV" or "Bedmap3"
}

interface IrhIndex {
  datasets: Record<string, { vertices: string; runs: string }>;
  features: Feature[];
}

// Young (yellow) to old (purple)
const AGE_COLORS = ["#fde725", "#5ec962", "#21918c", "#3b528b", "#440154"];

export function ageColor(ageKa: number | null, maxAgeKa: number, target = new THREE.Color()): THREE.Color {
  if (ageKa === null) return target.set("#d0d0d0");
  const t = THREE.MathUtils.clamp(ageKa / maxAgeKa, 0, 1) * (AGE_COLORS.length - 1);
  const i = Math.min(Math.floor(t), AGE_COLORS.length - 2);
  return target.set(AGE_COLORS[i]).lerp(new THREE.Color(AGE_COLORS[i + 1]), t - i);
}

export function ageGradientCss(): string {
  return `linear-gradient(to right, ${AGE_COLORS.join(", ")})`;
}

/** Segment endpoints (start, end) of every run, with the feature of each segment. */
function segments(vertices: Float32Array, runs: Uint32Array) {
  let n = 0;
  for (let r = 0; r < runs.length; r += 3) n += runs[r + 2] - 1;
  const positions = new Float32Array(n * 6);
  const features = new Uint32Array(n);
  let s = 0;
  for (let r = 0; r < runs.length; r += 3) {
    const [feature, first, count] = [runs[r], runs[r + 1], runs[r + 2]];
    for (let v = first; v < first + count - 1; v++, s++) {
      positions.set(vertices.subarray(v * 3, v * 3 + 6), s * 6);
      features[s] = feature;
    }
  }
  return { positions, features };
}

const CELL = 16; // px, screen grid cell size

/** Segments projected to the screen, binned by grid cell. */
interface ScreenIndex {
  width: number;
  height: number;
  columns: number;
  // Start and end of each segment: x, y (px) and depth (NDC)
  sx: Float32Array;
  sy: Float32Array;
  depth: Float32Array;
  // Segments of cell c: items[cellStart[c]] ... items[cellStart[c + 1] - 1]
  cellStart: Uint32Array;
  items: Uint32Array;
}

export class Isochrones {
  readonly group = new THREE.Group();
  features: Feature[] = [];
  maxAgeKa = 1;

  private material = new LineMaterial({ linewidth: 2.5, vertexColors: true });
  // All segments: positions (start xyz, end xyz) and feature of each
  private parts: { positions: Float32Array; features: Uint32Array }[] = [];
  private positions = new Float32Array(0);
  private segmentFeatures = new Uint32Array(0);
  private screenIndex: ScreenIndex | null = null;
  // Positions of the segments of each feature, for highlighting
  private featureSegments = new Map<number, Float32Array[]>();
  private highlight: LineSegments2 | null = null;
  private highlightMaterial = new LineMaterial({
    linewidth: 5,
    color: 0xff4d4d,
    depthTest: false,
    transparent: true,
  });

  async load(baseUrl: string): Promise<void> {
    const response = await fetch(`${baseUrl}/irh.json`);
    if (!response.ok) {
      throw new Error(
        `${baseUrl}/irh.json not found. Export it with: python -m anta_database.web.irh /path/to/AntADatabase --bedmap bedmap3.nc`,
      );
    }
    const index: IrhIndex = await response.json();
    this.features = index.features;
    this.maxAgeKa = Math.max(1, ...this.features.map((f) => f.age_ka ?? 0));

    await Promise.all(
      Object.values(index.datasets).map(async (files) => {
        const [vertices, runs] = await Promise.all([
          fetchBytes(`${baseUrl}/${files.vertices}`).then((b) => new Float32Array(b)),
          fetchBytes(`${baseUrl}/${files.runs}`).then((b) => new Uint32Array(b)),
        ]);
        this.addDataset(vertices, runs);
      }),
    );
    this.positions = new Float32Array(this.parts.reduce((n, p) => n + p.positions.length, 0));
    this.segmentFeatures = new Uint32Array(this.positions.length / 6);
    let offset = 0;
    for (const part of this.parts) {
      this.positions.set(part.positions, offset * 6);
      this.segmentFeatures.set(part.features, offset);
      offset += part.features.length;
    }
    this.parts = [];
    this.invalidate();
  }

  private addDataset(vertices: Float32Array, runs: Uint32Array): void {
    const { positions, features } = segments(vertices, runs);
    const colors = new Float32Array(positions.length);
    const color = new THREE.Color();
    for (let s = 0; s < features.length; s++) {
      const f = features[s];
      ageColor(this.features[f].age_ka, this.maxAgeKa, color);
      colors.set([color.r, color.g, color.b, color.r, color.g, color.b], s * 6);
      const list = this.featureSegments.get(f) ?? [];
      list.push(positions.subarray(s * 6, s * 6 + 6));
      this.featureSegments.set(f, list);
    }

    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(positions);
    geometry.setColors(colors);
    const lines = new LineSegments2(geometry, this.material);
    lines.frustumCulled = false;
    this.group.add(lines);
    this.parts.push({ positions, features });
  }

  setExaggeration(exaggeration: number): void {
    this.group.scale.z = exaggeration / 1000;
    this.invalidate();
  }

  /** Call when the view changed: the screen index is rebuilt on the next pick. */
  invalidate(): void {
    this.screenIndex = null;
  }

  /** Canvas size in CSS pixels, so line widths are in CSS pixels. */
  setResolution(width: number, height: number): void {
    for (const material of [this.material, this.highlightMaterial]) {
      material.resolution.set(width, height);
    }
  }

  setLineWidth(width: number): void {
    this.material.linewidth = width;
    this.highlightMaterial.linewidth = width + 2.5;
  }

  private buildScreenIndex(camera: THREE.Camera, width: number, height: number): ScreenIndex {
    this.group.updateMatrixWorld();
    camera.updateMatrixWorld();
    const m = new THREE.Matrix4()
      .multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse)
      .multiply(this.group.matrixWorld).elements;
    const p = this.positions;
    const n = p.length / 3; // segment ends
    // Clip coordinates of every segment end
    const clip = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      const x = p[3 * i];
      const y = p[3 * i + 1];
      const z = p[3 * i + 2];
      clip[4 * i] = m[0] * x + m[4] * y + m[8] * z + m[12];
      clip[4 * i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
      clip[4 * i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
      clip[4 * i + 3] = m[3] * x + m[7] * y + m[11] * z + m[15];
    }
    // Segments crossing the near plane are cut there rather than dropped
    const near = 1e-3;
    const sx = new Float32Array(n);
    const sy = new Float32Array(n);
    const depth = new Float32Array(n);
    const front = new Uint8Array(n);
    for (let a = 0; a < n; a += 2) {
      const b = a + 1;
      const wa = clip[4 * a + 3];
      const wb = clip[4 * b + 3];
      if (wa < near && wb < near) continue; // behind the camera
      for (const [i, j, wi, wj] of [
        [a, b, wa, wb],
        [b, a, wb, wa],
      ]) {
        if (wi < near) {
          const t = (near - wi) / (wj - wi);
          for (let k = 0; k < 4; k++) clip[4 * i + k] += t * (clip[4 * j + k] - clip[4 * i + k]);
        }
      }
      for (const i of [a, b]) {
        const w = clip[4 * i + 3];
        sx[i] = (clip[4 * i] / w + 1) * 0.5 * width;
        sy[i] = (1 - clip[4 * i + 1] / w) * 0.5 * height;
        depth[i] = clip[4 * i + 2] / w;
        front[i] = 1;
      }
    }

    // Cells covered by the bounding box of each segment on screen, counted then filled
    const columns = Math.ceil(width / CELL);
    const rows = Math.ceil(height / CELL);
    const cellStart = new Uint32Array(columns * rows + 1);
    const segments = n / 2;
    const box = new Int32Array(segments * 4).fill(-1);
    for (let s = 0; s < segments; s++) {
      const a = 2 * s;
      const b = a + 1;
      if (!front[a] || !front[b]) continue;
      const c0 = Math.max(0, Math.floor(Math.min(sx[a], sx[b]) / CELL));
      const c1 = Math.min(columns - 1, Math.floor(Math.max(sx[a], sx[b]) / CELL));
      const r0 = Math.max(0, Math.floor(Math.min(sy[a], sy[b]) / CELL));
      const r1 = Math.min(rows - 1, Math.floor(Math.max(sy[a], sy[b]) / CELL));
      if (c0 > c1 || r0 > r1) continue; // off screen
      box.set([c0, c1, r0, r1], s * 4);
      for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) cellStart[r * columns + c + 1]++;
    }
    for (let c = 0; c < columns * rows; c++) cellStart[c + 1] += cellStart[c];
    const items = new Uint32Array(cellStart[columns * rows]);
    const fill = cellStart.slice(0, columns * rows);
    for (let s = 0; s < segments; s++) {
      if (box[s * 4] < 0) continue;
      const [c0, c1, r0, r1] = box.subarray(s * 4, s * 4 + 4);
      for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) items[fill[r * columns + c]++] = s;
    }
    return { width, height, columns, sx, sy, depth, cellStart, items };
  }

  /**
   * Feature of the line closest to a pixel (CSS pixels from the top-left of the
   * canvas), within radius pixels, or null.
   */
  pick(camera: THREE.Camera, x: number, y: number, width: number, height: number, radius = 8): number | null {
    if (!this.group.visible || this.segmentFeatures.length === 0) return null;
    let index = this.screenIndex;
    if (!index || index.width !== width || index.height !== height) {
      index = this.screenIndex = this.buildScreenIndex(camera, width, height);
    }
    const { columns, sx, sy, depth, cellStart, items } = index;
    const rows = (cellStart.length - 1) / columns;
    const c0 = Math.max(0, Math.floor((x - radius) / CELL));
    const c1 = Math.min(columns - 1, Math.floor((x + radius) / CELL));
    const r0 = Math.max(0, Math.floor((y - radius) / CELL));
    const r1 = Math.min(rows - 1, Math.floor((y + radius) / CELL));

    let best = -1;
    let bestDistance = radius;
    let bestDepth = Infinity;
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const cell = r * columns + c;
        for (let k = cellStart[cell]; k < cellStart[cell + 1]; k++) {
          const s = items[k];
          const a = 2 * s;
          const b = a + 1;
          // Closest point of the segment to the cursor
          const dx = sx[b] - sx[a];
          const dy = sy[b] - sy[a];
          const length2 = dx * dx + dy * dy;
          const t = length2 > 0 ? THREE.MathUtils.clamp(((x - sx[a]) * dx + (y - sy[a]) * dy) / length2, 0, 1) : 0;
          const distance = Math.hypot(sx[a] + t * dx - x, sy[a] + t * dy - y);
          if (distance > radius) continue;
          const d = depth[a] + t * (depth[b] - depth[a]);
          // Clearly closer on screen, or about as close and in front
          const closer = distance < bestDistance - 1.5;
          const similar = Math.abs(distance - bestDistance) <= 1.5;
          if (best < 0 || closer || (similar && d < bestDepth)) {
            best = s;
            bestDistance = distance;
            bestDepth = d;
          }
        }
      }
    }
    return best < 0 ? null : this.segmentFeatures[best];
  }

  /** Draw a feature on top of everything, or nothing with null. */
  setHighlight(feature: number | null): void {
    if (this.highlight) {
      this.group.remove(this.highlight);
      this.highlight.geometry.dispose();
      this.highlight = null;
    }
    const parts = feature === null ? undefined : this.featureSegments.get(feature);
    if (!parts) return;
    const positions = new Float32Array(parts.length * 6);
    parts.forEach((p, i) => positions.set(p, i * 6));
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(positions);
    this.highlight = new LineSegments2(geometry, this.highlightMaterial);
    this.highlight.frustumCulled = false;
    this.highlight.renderOrder = 10;
    this.group.add(this.highlight);
  }
}
