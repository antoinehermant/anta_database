/**
 * The outlines of the regions chosen in the filter: a place as the circle of its radius (the lines
 * within it are at the place), an ice sheet or basin by its outline (regions_outlines.json.gz of
 * the export, anta_database.web.regions).
 *
 * Drawn like the ice divides (outlines.ts): a black dashed line on the ice surface, and a thin
 * transparent wall from it down to the bed through the ice. One feature per region in the
 * lines, added the first time it is chosen, the others hidden; the walls are those of the regions
 * shown. Hovered like the other lines (main.ts). Heights come from the terrain the viewer draws; over the sea, the surface at sea level.
 *
 * A blue flag with the name of the region on a pole at its middle (anta_json #24), like the red flags
 * of the ice cores (icecores.ts): a sprite of a fixed size in pixels, smaller for smaller regions.
 */
import * as THREE from "three";
import { LineLayer } from "./lines";
import type { HeightAt } from "./outlines";
import type { Region } from "./irh";

const COLOR = "#000000";
const STEP = 2; // km, at most between two vertices, so the lines follow the terrain
const LIFT = 25; // m above the surface, so the line is not hidden in the terrain between its vertices
const DASH: [number, number] = [8, 6]; // px
const FLAG_COLOR = "#2a78d6";
// Flag height (px): FLAG_PX_MIN for a region of FLAG_SMALL_KM of radius or less, up to FLAG_PX_MAX (that of the
// ice cores) for the large ones, growing with the logarithm of the radius
const [FLAG_PX_MIN, FLAG_PX_MAX, FLAG_SMALL_KM] = [12, 22, 20];
const POLE_PX = 1.5; // width
const POLE_PX_HEIGHT = 40; // for the largest flags, in proportion for the others

function flagPixels(radius: number): number {
  return Math.min(FLAG_PX_MAX, Math.max(FLAG_PX_MIN, FLAG_PX_MIN + 2.5 * Math.log2(Math.max(radius, 1) / FLAG_SMALL_KM)));
}

function flagTexture(label: string, px: number): { texture: THREE.CanvasTexture; aspect: number } {
  const scale = 3; // sharp on high density screens
  const height = px * scale;
  const font = `600 ${Math.round(px * 0.6 * scale)}px system-ui, sans-serif`;
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d")!;
  context.font = font;
  const notch = Math.round(px * 0.23 * scale);
  const pad = Math.round(px * 0.23 * scale);
  const width = Math.ceil(context.measureText(label).width + 2 * pad + notch);
  canvas.width = width;
  canvas.height = height;
  context.font = font;
  context.fillStyle = FLAG_COLOR;
  context.beginPath();
  context.moveTo(0, 0);
  context.lineTo(width, 0);
  context.lineTo(width - notch, height / 2);
  context.lineTo(width, height);
  context.lineTo(0, height);
  context.closePath();
  context.fill();
  context.fillStyle = "#ffffff";
  context.textBaseline = "middle";
  context.fillText(label, pad, height / 2 + scale);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return { texture, aspect: width / height };
}

/** The middle of a region (x, y km) and its radius (km): a place, its circle; an outline, the centroid and
 * the radius of the circle of the same area. */
function middle(region: Region, outlines: Record<string, number[][]>): [number, number, number] {
  const [xmin, xmax, ymin, ymax] = region.box.map((v) => v / 1000);
  const box: [number, number, number] = [(xmin + xmax) / 2, (ymin + ymax) / 2, Math.max(xmax - xmin, ymax - ymin) / 2];
  if (region.kind === "place") return [box[0], box[1], region.radius_km ?? box[2]];
  let [area, cx, cy] = [0, 0, 0];
  for (const ring of rings(region, outlines)) {
    for (let i = 0; i + 3 < ring.length; i += 2) {
      const cross = ring[i] * ring[i + 3] - ring[i + 2] * ring[i + 1];
      area += cross / 2;
      cx += ((ring[i] + ring[i + 2]) * cross) / 6;
      cy += ((ring[i + 1] + ring[i + 3]) * cross) / 6;
    }
  }
  if (!area) return box;
  return [cx / area, cy / area, Math.sqrt(Math.abs(area) / Math.PI)];
}

interface Flag {
  x: number;
  y: number;
  height: number; // surface elevation (m)
  px: number; // flag height (px)
  aspect: number;
  pole: THREE.Mesh;
  flag: THREE.Sprite;
}

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
  // Flags are drawn last, over everything, as those of the ice cores: in their own group
  private flagGroup = new THREE.Group();
  private flags = new Map<Region, Flag>();
  private exaggeration = 1;
  private pole = new THREE.CylinderGeometry(1, 1, 1, 12, 1).rotateX(Math.PI / 2).translate(0, 0, 0.5);
  private poleMaterial = new THREE.MeshBasicMaterial({ color: "#3a3f47" });
  // The region of each feature
  private regions: Region[] = [];

  constructor() {
    this.walls.frustumCulled = false;
    this.walls.renderOrder = 2; // after the transparent ice surface, which they cross
    this.surface.group.renderOrder = 3;
    this.flagGroup.renderOrder = 100;
    this.group.add(this.walls, this.surface.group, this.flagGroup);
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
    for (const region of regions) {
      if (this.flags.has(region)) continue;
      const [x, y, radius] = middle(region, outlines);
      const px = flagPixels(radius);
      const { texture, aspect } = flagTexture(region.name, px);
      const flag = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: texture, sizeAttenuation: false, depthTest: false, depthWrite: false, alphaTest: 0.5, toneMapped: false }),
      );
      flag.center.set(0, 0); // the bottom left corner at the top of the pole
      const pole = new THREE.Mesh(this.pole, this.poleMaterial);
      for (const object of [flag, pole]) object.frustumCulled = false;
      const height = heightAt("surface", x, y) ?? 0;
      this.flags.set(region, { x, y, height, px, aspect, pole, flag });
      this.flagGroup.add(pole, flag);
    }
    const chosen = new Set(regions);
    for (const [region, f] of this.flags) f.pole.visible = f.flag.visible = chosen.has(region);
  }

  /** Sizes of the flags for the current view: called before each render. */
  update(camera: THREE.PerspectiveCamera, height: number): void {
    const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const position = new THREE.Vector3();
    for (const f of this.flags.values()) {
      if (!f.flag.visible) continue;
      const top = (f.height * this.exaggeration) / 1000;
      const pixel = (2 * tan * camera.position.distanceTo(position.set(f.x, f.y, top))) / height; // km
      const poleHeight = POLE_PX_HEIGHT * (f.px / FLAG_PX_MAX) * pixel;
      const radius = (POLE_PX / 2) * pixel;
      f.pole.position.set(f.x, f.y, top);
      f.pole.scale.set(radius, radius, poleHeight);
      f.flag.position.set(f.x, f.y, top + poleHeight);
      const flagHeight = (2 * f.px) / height; // sprites without size attenuation: clip space (2 = screen height)
      f.flag.scale.set(flagHeight * f.aspect * tan, flagHeight * tan, 1);
    }
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
    this.exaggeration = exaggeration;
    this.walls.scale.z = exaggeration / 1000;
    for (const layer of this.lineLayers) layer.setExaggeration(exaggeration);
  }

  /** Dashes of the same length on screen at any zoom, from the size of a pixel (km). */
  setDashPixel(pixel: number): void {
    for (const layer of this.lineLayers) layer.setDashPixel(pixel, ...DASH);
  }
}
