/**
 * 3D polylines exported by anta_database.web.irh, drawn as wide line segments.
 *
 * Vertices are rows of (x km, y km, elevation m, ...) packed per run, with runs
 * of (feature id, first vertex, vertex count). The group is scaled in z by
 * exaggeration / 1000, like the terrain. The segments of a feature are
 * contiguous, so a feature is a range of segments.
 *
 * Hover works in screen space: once the view is still, all segments are
 * projected to the screen and binned in a grid of small cells. The hovered
 * line is the closest one within a few pixels of the cursor, the nearest to
 * the camera when several are about as close.
 *
 * Features can be hidden (filters): each segment has a hidden flag, checked in
 * the vertex shader, so filtering never rebuilds the geometry.
 *
 * The hovered feature is drawn on top of everything, with a bright white glow around it.
 * Selected features (the line of the open profile) are drawn again, wider and with a fainter
 * golden glow, after the ice surface so they show through it; the others can be dimmed:
 * thinner and partly transparent.
 */
import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

/** Lines of one file (dataset): segments first ... first + count - 1 of the layer. */
export interface Part {
  name: string;
  vertices: Float32Array;
  stride: number; // values per vertex
  // Vertex (in vertices) of the start and end of each segment
  ends: Uint32Array;
  first: number;
  count: number;
  lines: LineSegments2;
}

export interface Hit {
  feature: number;
  segment: number;
  end: 0 | 1; // the end of the segment closest to the cursor
  distance: number; // px
  depth: number; // NDC, smaller is closer
}

const CELL = 16; // px, screen grid cell size
// How much wider the glow of a selected line is than the line (px)
const SELECTION_GLOW = 5;
// Width of the other lines while some are selected: DIMMED_WIDTH of the default width, and
// DIMMED_RATE px wider for each px wider in the menu (1: as the selected lines, 0: never wider)
const DIMMED_WIDTH = 0.6;
const DIMMED_RATE = 0.3;

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

export class LineLayer {
  readonly group = new THREE.Group();
  readonly material: LineMaterial;
  parts: Part[] = [];
  // All segments: positions (start xyz, end xyz) and feature of each
  private positions = new Float32Array(0);
  private segmentFeatures = new Uint32Array(0);
  // Segments of the parts added since the last finish()
  private pending: { positions: Float32Array; features: Uint32Array }[] = [];
  // Segments of feature f: featureFirst[f] ... featureLast[f]
  private featureFirst = new Int32Array(0);
  private featureLast = new Int32Array(0);
  // 1 for hidden features
  private hiddenFeatures = new Uint8Array(0);
  private screenIndex: ScreenIndex | null = null;
  // The hovered feature: its glow, then the line over it
  private highlight: LineSegments2[] = [];
  private highlightGlow: LineMaterial;
  private highlightMaterial: LineMaterial;
  // Selected features, drawn in this group (in the layer group, after the ice surface)
  private selected: number[] = [];
  private selectedSet = new Set<number>();
  private readonly selection = new THREE.Group();
  private selectionGlow: LineMaterial;
  private selectionMaterial: LineMaterial;
  // Width and opacity of the lines, and whether they are dimmed for a selection
  private width: number;
  // The width it was made with: dimmed lines grow slower from it (DIMMED_RATE)
  private readonly baseWidth: number;
  private readonly opacity: number;
  private readonly transparent: boolean;
  private dimmed = false;

  constructor(options: { linewidth: number; color?: THREE.ColorRepresentation; transparent?: boolean; dashed?: boolean }) {
    const transparent = options.transparent ?? false;
    this.material = new LineMaterial({
      linewidth: options.linewidth,
      vertexColors: options.color === undefined,
      color: options.color ?? 0xffffff,
      transparent,
      dashed: options.dashed ?? false,
    });
    // Hidden segments are moved out of the view volume, so they draw nothing
    this.material.onBeforeCompile = (shader) => {
      shader.vertexShader = `attribute float instanceHidden;\n${shader.vertexShader}`.replace(
        "#include <fog_vertex>",
        "#include <fog_vertex>\n\tif (instanceHidden > 0.5) gl_Position = vec4(0.0, 0.0, 2.0, 1.0);",
      );
    };
    this.highlightGlow = new LineMaterial({ linewidth: options.linewidth + 6, color: 0xffffff, depthTest: false, transparent: true, opacity: 0.7, depthWrite: false });
    this.highlightMaterial = new LineMaterial({
      linewidth: options.linewidth + 1,
      vertexColors: options.color === undefined,
      color: options.color ?? 0xffffff,
      depthTest: false,
      transparent: true,
    });
    this.width = this.baseWidth = options.linewidth;
    this.opacity = this.material.opacity;
    this.transparent = transparent;
    // Transparent (even when opaque) so it is drawn with the transparent things, after the
    // surface, which is drawn first by the order of its group
    this.selectionGlow = new LineMaterial({ linewidth: options.linewidth + SELECTION_GLOW, color: 0xffd36b, transparent: true, opacity: 0.45, depthWrite: false });
    this.selectionMaterial = new LineMaterial({
      linewidth: options.linewidth + 1.5,
      vertexColors: options.color === undefined,
      color: options.color ?? 0xffffff,
      transparent: true,
    });
    this.selection.renderOrder = 5;
    this.group.add(this.selection);
  }

  get loaded(): boolean {
    return this.segmentFeatures.length > 0;
  }

  /** Add the lines of a file. Call finish() once all are added. */
  add(name: string, vertices: Float32Array, stride: number, runs: Uint32Array): Part {
    let n = 0;
    for (let r = 0; r < runs.length; r += 3) n += runs[r + 2] - 1;
    const ends = new Uint32Array(n * 2);
    const features = new Uint32Array(n);
    const positions = new Float32Array(n * 6);
    let s = 0;
    for (let r = 0; r < runs.length; r += 3) {
      const [feature, first, count] = [runs[r], runs[r + 1], runs[r + 2]];
      for (let v = first; v < first + count - 1; v++, s++) {
        ends[2 * s] = v;
        ends[2 * s + 1] = v + 1;
        features[s] = feature;
        for (let k = 0; k < 2; k++) {
          positions.set(vertices.subarray((v + k) * stride, (v + k) * stride + 3), s * 6 + k * 3);
        }
      }
    }
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(positions);
    if (this.material.vertexColors) geometry.setColors(new Float32Array(positions.length));
    if (this.material.dashed) {
      // Distance (km) along each run at the segment ends, horizontally: dashes
      // don't depend on the exaggeration
      const distances = new Float32Array(n * 2);
      for (let k = 0; k < n; k++) {
        const continued = k > 0 && ends[2 * k] === ends[2 * k - 1];
        distances[2 * k] = continued ? distances[2 * k - 1] : 0;
        const dx = positions[6 * k + 3] - positions[6 * k];
        const dy = positions[6 * k + 4] - positions[6 * k + 1];
        distances[2 * k + 1] = distances[2 * k] + Math.hypot(dx, dy);
      }
      const buffer = new THREE.InstancedInterleavedBuffer(distances, 2, 1);
      geometry.setAttribute("instanceDistanceStart", new THREE.InterleavedBufferAttribute(buffer, 1, 0));
      geometry.setAttribute("instanceDistanceEnd", new THREE.InterleavedBufferAttribute(buffer, 1, 1));
    }
    geometry.setAttribute("instanceHidden", new THREE.InstancedBufferAttribute(new Float32Array(n), 1));
    const lines = new LineSegments2(geometry, this.material);
    lines.frustumCulled = false;
    this.group.add(lines);
    const part: Part = { name, vertices, stride, ends, first: 0, count: n, lines };
    this.parts.push(part);
    this.pending.push({ positions, features });
    return part;
  }

  /** Gather the segments of all parts, for picking and highlighting. */
  finish(): void {
    const total = this.parts.reduce((n, p) => n + p.count, 0);
    this.positions = new Float32Array(total * 6);
    this.segmentFeatures = new Uint32Array(total);
    let offset = 0;
    let maxFeature = -1;
    this.parts.forEach((part, i) => {
      const { positions, features } = this.pending[i];
      part.first = offset;
      this.positions.set(positions, offset * 6);
      this.segmentFeatures.set(features, offset);
      for (const f of features) maxFeature = Math.max(maxFeature, f);
      offset += part.count;
    });
    this.pending = [];
    this.featureFirst = new Int32Array(maxFeature + 1).fill(-1);
    this.featureLast = new Int32Array(maxFeature + 1).fill(-1);
    this.hiddenFeatures = new Uint8Array(maxFeature + 1);
    for (let s = 0; s < total; s++) {
      const f = this.segmentFeatures[s];
      if (this.featureFirst[f] < 0) this.featureFirst[f] = s;
      this.featureLast[f] = s;
    }
    this.invalidate();
  }

  /** Part and vertex of a segment end. */
  vertexOf(segment: number, end: 0 | 1): { part: Part; vertex: number } {
    let lo = 0;
    let hi = this.parts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.parts[mid].first <= segment) lo = mid;
      else hi = mid - 1;
    }
    const part = this.parts[lo];
    return { part, vertex: part.ends[2 * (segment - part.first) + end] };
  }

  /** Elevation (m) of a segment end. */
  elevation(segment: number, end: 0 | 1): number {
    return this.positions[segment * 6 + end * 3 + 2];
  }

  /** Number of features (ids from 0). */
  get featureCount(): number {
    return this.featureFirst.length;
  }

  isHidden(feature: number): boolean {
    return this.hiddenFeatures[feature] === 1;
  }

  /** Segments of a feature: start xyz, end xyz of each, or null if it has none. */
  featureSegments(feature: number): Float32Array | null {
    if (feature >= this.featureFirst.length || this.featureFirst[feature] < 0) return null;
    return this.positions.subarray(this.featureFirst[feature] * 6, (this.featureLast[feature] + 1) * 6);
  }

  /** Feature of a segment. */
  feature(segment: number): number {
    return this.segmentFeatures[segment];
  }

  /**
   * Set the colour of every segment end of a part, from the vertex and the
   * feature; the colour buffers are updated in place.
   */
  recolor(part: Part, colorOf: (vertex: number, feature: number, target: THREE.Color) => THREE.Color): void {
    const attribute = part.lines.geometry.getAttribute("instanceColorStart") as THREE.InterleavedBufferAttribute;
    const buffer = attribute.data;
    const colors = buffer.array as Float32Array;
    const color = new THREE.Color();
    for (let s = 0; s < part.count; s++) {
      const feature = this.segmentFeatures[part.first + s];
      for (let k = 0; k < 2; k++) {
        colorOf(part.ends[2 * s + k], feature, color);
        colors[s * 6 + k * 3] = color.r;
        colors[s * 6 + k * 3 + 1] = color.g;
        colors[s * 6 + k * 3 + 2] = color.b;
      }
    }
    buffer.needsUpdate = true;
    if (this.selected.length) this.drawSelection();
  }

  /** Hide the features for which hidden(feature) is true, and show the others. */
  setHidden(hidden: (feature: number) => boolean): void {
    for (let f = 0; f < this.hiddenFeatures.length; f++) this.hiddenFeatures[f] = hidden(f) ? 1 : 0;
    for (const part of this.parts) {
      const attribute = part.lines.geometry.getAttribute("instanceHidden") as THREE.InstancedBufferAttribute;
      const flags = attribute.array as Float32Array;
      let shown = 0;
      for (let s = 0; s < part.count; s++) {
        flags[s] = this.hiddenFeatures[this.segmentFeatures[part.first + s]];
        shown += 1 - flags[s];
      }
      attribute.needsUpdate = true;
      // Nothing to draw at all for a part (dataset) fully hidden
      part.lines.visible = shown > 0;
    }
    if (this.selected.length) this.drawSelection();
  }

  /**
   * Draw these features out (the line of the open profile), and dim the others with dim;
   * nothing selected and nothing dimmed with an empty list.
   */
  setSelection(features: number[], dim = false): void {
    this.selected = features;
    this.selectedSet = new Set(features);
    this.dimmed = dim && features.length > 0;
    this.applyStyle();
    this.drawSelection();
  }

  private drawSelection(): void {
    // The glow and the line share their geometry
    (this.selection.children[0] as LineSegments2 | undefined)?.geometry.dispose();
    this.selection.clear();
    const geometry = this.geometryOf(this.selected);
    if (!geometry) return;
    // The glow, then the line over it
    [this.selectionGlow, this.selectionMaterial].forEach((material, i) => {
      const lines = new LineSegments2(geometry, material);
      lines.frustumCulled = false;
      lines.renderOrder = i;
      this.selection.add(lines);
    });
  }

  /** Segments of the features shown, with their colours, or null if none has any. */
  private geometryOf(features: number[]): LineSegmentsGeometry | null {
    const positions: Float32Array[] = [];
    const colors: Float32Array[] = [];
    for (const f of features) {
      if (this.isHidden(f)) continue;
      const segments = this.featureSegments(f);
      if (!segments) continue;
      positions.push(segments);
      if (this.material.vertexColors) {
        // The colours of the feature, from its part: its segments are contiguous in one part
        const { part } = this.vertexOf(this.featureFirst[f], 0);
        const attribute = part.lines.geometry.getAttribute("instanceColorStart") as THREE.InterleavedBufferAttribute;
        const start = (this.featureFirst[f] - part.first) * 6;
        colors.push((attribute.data.array as Float32Array).slice(start, start + segments.length));
      }
    }
    if (!positions.length) return null;
    const join = (arrays: Float32Array[]) => {
      const all = new Float32Array(arrays.reduce((n, a) => n + a.length, 0));
      let offset = 0;
      for (const a of arrays) {
        all.set(a, offset);
        offset += a.length;
      }
      return all;
    };
    const geometry = new LineSegmentsGeometry();
    geometry.setPositions(join(positions));
    if (colors.length) geometry.setColors(join(colors));
    return geometry;
  }

  /** Width and opacity of the lines: thinner and partly transparent when dimmed. */
  private applyStyle(): void {
    const material = this.material;
    // Dimmed: thinner, and wider lines in the menu widen them less than the selected ones, so
    // these still stand out
    const dimmedWidth = DIMMED_WIDTH * this.baseWidth + DIMMED_RATE * (this.width - this.baseWidth);
    material.linewidth = this.dimmed ? Math.max(1, dimmedWidth) : this.width;
    material.opacity = this.dimmed ? this.opacity * 0.45 : this.opacity;
    const transparent = this.transparent || this.dimmed;
    if (material.transparent !== transparent) {
      material.transparent = transparent;
      material.needsUpdate = true;
    }
    this.selectionMaterial.linewidth = this.width + 1.5;
    this.selectionGlow.linewidth = this.width + SELECTION_GLOW;
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
    for (const material of [this.material, this.highlightGlow, this.highlightMaterial, this.selectionGlow, this.selectionMaterial]) {
      material.resolution.set(width, height);
    }
  }

  /** Dashes and gaps of the same length on screen at any zoom, from the size of a pixel (km). */
  setDashPixel(pixel: number, dash = 7, gap = 5): void {
    this.material.dashSize = dash * pixel;
    this.material.gapSize = gap * pixel;
  }

  setLineWidth(width: number): void {
    this.width = width;
    this.applyStyle();
    this.highlightGlow.linewidth = width + 6;
    this.highlightMaterial.linewidth = width + 1;
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
   * Line closest to a pixel (CSS pixels from the top-left of the canvas),
   * within radius pixels, or null. Lines behind a selected one there are left out: the
   * selected lines are often packed in front of others, which can be pointed at once the
   * view turned so that they are no longer behind.
   */
  pick(camera: THREE.Camera, x: number, y: number, width: number, height: number, radius = 40): Hit | null {
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

    let best: Hit | null = null;
    let bestSelected: Hit | null = null;
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const cell = r * columns + c;
        for (let k = cellStart[cell]; k < cellStart[cell + 1]; k++) {
          const s = items[k];
          if (this.hiddenFeatures[this.segmentFeatures[s]]) continue;
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
          const feature = this.segmentFeatures[s];
          const selected = this.selectedSet.has(feature);
          if (selected ? !bestSelected || closer(distance, d, bestSelected) : !best || closer(distance, d, best)) {
            const hit: Hit = { feature, segment: s, end: t < 0.5 ? 0 : 1, distance, depth: d };
            if (selected) bestSelected = hit;
            else best = hit;
          }
        }
      }
    }
    if (!bestSelected) return best;
    // Another line only in front of the selected one
    return best && best.depth < bestSelected.depth && closer(best.distance, best.depth, bestSelected) ? best : bestSelected;
  }

  /** Draw a feature on top of everything, glowing, or nothing with null. */
  setHighlight(feature: number | null): void {
    if (this.highlight.length) {
      this.group.remove(...this.highlight);
      this.highlight[0].geometry.dispose();
      this.highlight = [];
    }
    const geometry = feature === null ? null : this.geometryOf([feature]);
    if (!geometry) return;
    // The glow, then the line over it in its colours
    this.highlight = [new LineSegments2(geometry, this.highlightGlow), new LineSegments2(geometry, this.highlightMaterial)];
    this.highlight.forEach((lines, i) => {
      lines.frustumCulled = false;
      lines.renderOrder = 10 + i;
      this.group.add(lines);
    });
  }
}

/** Whether a line at distance (px) and depth is a better hit: clearly closer on screen, or about as close and in front. */
export function closer(distance: number, depth: number, best: { distance: number; depth: number }): boolean {
  if (distance < best.distance - 1.5) return true;
  return Math.abs(distance - best.distance) <= 1.5 && depth < best.depth;
}
