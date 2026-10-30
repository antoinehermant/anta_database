/**
 * MEaSUREs ice velocities exported by anta_database.web.velocity: a speed
 * texture to colour the ice surface, and flow lines moving with the ice.
 *
 * Flow lines are short streaks, transparent at the tail, advected on the CPU
 * through the velocity field and lifted to the ice surface in the vertex
 * shader. They live around the centre of the view, so their density on screen
 * doesn't depend on the zoom, and move at a speed growing with the log of the
 * ice speed, like the colours: slow ice still visibly moves.
 */
import * as THREE from "three";
import { fetchBytes, fetchData, gridOf, type Grid, type Level, type TerrainUniforms } from "./terrain";

export interface VelocityMeta {
  nodata: number;
  citation: string;
  doi: string;
  grids: { speed: Level; vectors: Level };
}

// Range of the colours and flow line speeds, log10(m/a)
export const LOG_SPEED_RANGE = new THREE.Vector2(0, Math.log10(3000));

export async function loadVelocityMeta(baseUrl: string): Promise<VelocityMeta> {
  const response = await fetchData(`${baseUrl}/velocity.json`);
  if (!response.ok) {
    throw new Error(
      `${baseUrl}/velocity.json not found. Export it with: python -m anta_database.web.velocity antarctica_ice_velocity_450m_v2.nc`,
    );
  }
  return response.json();
}

/** Speed texture: R = speed (m/a), G = 1 where there is data. */
export async function loadSpeedTexture(baseUrl: string, meta: VelocityMeta): Promise<THREE.DataTexture> {
  const level = meta.grids.speed;
  const raw = new Int16Array(await fetchBytes(`${baseUrl}/${level.layers.speed.file}`));
  const data = new Uint16Array(raw.length * 2);
  const one = THREE.DataUtils.toHalfFloat(1);
  // Speeds are whole m/a: convert each value once
  const halves = new Uint16Array(32768);
  for (let v = 0; v < halves.length; v++) halves[v] = THREE.DataUtils.toHalfFloat(v);
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] === meta.nodata || raw[i] < 0) continue;
    data[2 * i] = halves[raw[i]];
    data[2 * i + 1] = one;
  }
  const texture = new THREE.DataTexture(data, level.nx, level.ny, THREE.RGFormat, THREE.HalfFloatType);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

/** Velocity components on a grid, sampled on the CPU. */
export class FlowField {
  private constructor(
    readonly grid: Grid,
    private vx: Int16Array,
    private vy: Int16Array,
    private nodata: number,
  ) {}

  static async load(baseUrl: string, meta: VelocityMeta): Promise<FlowField> {
    const level = meta.grids.vectors;
    const [vx, vy] = await Promise.all(
      ["vx", "vy"].map(async (name) => new Int16Array(await fetchBytes(`${baseUrl}/${level.layers[name].file}`))),
    );
    return new FlowField(gridOf(level), vx, vy, meta.nodata);
  }

  /** Velocity (m/a) at a point (km), bilinear between the valid cells around it, or false. */
  sample(x: number, y: number, out: THREE.Vector2): boolean {
    const { nx, ny, dx, x0, y0 } = this.grid;
    const fi = (x - x0) / dx;
    const fj = (y - y0) / dx;
    const i = Math.floor(fi);
    const j = Math.floor(fj);
    if (i < 0 || j < 0 || i >= nx - 1 || j >= ny - 1) return false;
    const u = fi - i;
    const v = fj - j;
    let sx = 0;
    let sy = 0;
    let sw = 0;
    for (const [di, dj, w] of [
      [0, 0, (1 - u) * (1 - v)],
      [1, 0, u * (1 - v)],
      [0, 1, (1 - u) * v],
      [1, 1, u * v],
    ]) {
      const k = (j + dj) * nx + i + di;
      if (this.vx[k] === this.nodata) continue;
      sx += w * this.vx[k];
      sy += w * this.vy[k];
      sw += w;
    }
    if (sw < 0.5) return false;
    out.set(sx / sw, sy / sw);
    return true;
  }
}

const vertexShader = /* glsl */ `
// One streak per instance, from its tail (seg.xy, km) to its head (seg.zw),
// as a quad of constant width on screen, at the height of the ice surface.
uniform sampler2D heightTex;
uniform vec2 texOrigin;
uniform float texDx;
uniform vec2 texSize;
uniform float exaggeration;
uniform vec2 resolution; // CSS px
uniform float width; // CSS px
attribute vec4 seg;
attribute float alpha;
varying float vAlpha;

vec4 project(vec2 p, out float valid) {
  vec2 uv = (p - texOrigin) / (texDx * texSize) + 0.5 / texSize;
  vec2 h = texture(heightTex, uv).rg;
  valid = h.g;
  // Slightly above the surface, so the streaks are not hidden in it
  return projectionMatrix * modelViewMatrix * vec4(p, (h.r + 15.0) * exaggeration / 1000.0, 1.0);
}

void main() {
  float validA;
  float validB;
  vec4 a = project(seg.xy, validA);
  vec4 b = project(seg.zw, validB);
  vAlpha = alpha * position.x * step(0.5, min(validA, validB));
  if (a.w <= 0.0 || b.w <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // behind the camera: clipped
    return;
  }
  vec2 sa = a.xy / a.w * resolution;
  vec2 sb = b.xy / b.w * resolution;
  vec2 dir = sb - sa;
  dir = length(dir) > 1e-6 ? normalize(dir) : vec2(1.0, 0.0);
  vec2 normal = vec2(-dir.y, dir.x);
  vec4 p = mix(a, b, position.x);
  p.xy += normal * position.y * 2.0 * width / resolution * p.w;
  gl_Position = p;
}
`;

const fragmentShader = /* glsl */ `
uniform vec3 color;
varying float vAlpha;

void main() {
  if (vAlpha < 0.01) discard;
  gl_FragColor = vec4(color, vAlpha);
  #include <colorspace_fragment>
}
`;

export class FlowLines {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.ShaderMaterial;
  private count: number;
  private segments: Float32Array;
  private alphas: Float32Array;
  // Head position (km), age and lifetime (s) of each streak
  private x: Float32Array;
  private y: Float32Array;
  private age: Float32Array;
  private life: Float32Array;
  private velocity = new THREE.Vector2();
  private centre = new THREE.Vector2();
  private radius = 1000;

  /**
   * surfaceUniforms: heightTex, texOrigin, texDx, texSize and exaggeration of
   * the ice surface layer, shared so the streaks follow its height.
   */
  constructor(
    private field: FlowField,
    surfaceUniforms: TerrainUniforms,
    count = 6000,
  ) {
    this.count = count;
    this.x = new Float32Array(count);
    this.y = new Float32Array(count);
    this.age = new Float32Array(count);
    this.life = new Float32Array(count);
    this.segments = new Float32Array(count * 4);
    this.alphas = new Float32Array(count);

    const geometry = new THREE.InstancedBufferGeometry();
    // Quad along the streak: x from tail (0) to head (1), y across (-0.5 to 0.5)
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([0, -0.5, 0, 1, -0.5, 0, 1, 0.5, 0, 0, 0.5, 0], 3),
    );
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    geometry.setAttribute("seg", new THREE.InstancedBufferAttribute(this.segments, 4).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("alpha", new THREE.InstancedBufferAttribute(this.alphas, 1).setUsage(THREE.DynamicDrawUsage));
    geometry.instanceCount = count;

    this.material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      transparent: true,
      depthWrite: false,
      uniforms: {
        heightTex: surfaceUniforms.heightTex,
        texOrigin: surfaceUniforms.texOrigin,
        texDx: surfaceUniforms.texDx,
        texSize: surfaceUniforms.texSize,
        exaggeration: surfaceUniforms.exaggeration,
        resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
        width: { value: 1.6 },
        color: { value: new THREE.Color("#15181d") },
      },
    });
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;
  }

  setResolution(width: number, height: number): void {
    this.material.uniforms.resolution.value.set(width, height);
  }

  /** Area (km) where the streaks live: a square around centre, of half-width radius. */
  setArea(centre: THREE.Vector2, radius: number): void {
    this.centre.copy(centre);
    this.radius = radius;
  }

  /** New streak at a random place with ice velocity; false if none was found. */
  private spawn(i: number): boolean {
    this.age[i] = 0;
    this.life[i] = 1.5 + 3 * Math.random();
    for (let attempt = 0; attempt < 8; attempt++) {
      const x = this.centre.x + (2 * Math.random() - 1) * this.radius;
      const y = this.centre.y + (2 * Math.random() - 1) * this.radius;
      if (this.field.sample(x, y, this.velocity)) {
        this.x[i] = x;
        this.y[i] = y;
        return true;
      }
    }
    return false;
  }

  /** Spread all streaks over the area again, at random stages of their lives. */
  respawn(): void {
    for (let i = 0; i < this.count; i++) {
      this.spawn(i);
      this.age[i] = Math.random() * this.life[i];
    }
    this.update(0);
  }

  /** Move the streaks by dt seconds (0 to only redraw them). */
  update(dt: number): void {
    const r = this.radius;
    for (let i = 0; i < this.count; i++) {
      this.age[i] += dt;
      const outside = Math.abs(this.x[i] - this.centre.x) > r || Math.abs(this.y[i] - this.centre.y) > r;
      let ok = this.age[i] < this.life[i] && !outside && this.field.sample(this.x[i], this.y[i], this.velocity);
      if (!ok) ok = this.spawn(i) && this.field.sample(this.x[i], this.y[i], this.velocity);
      if (!ok) {
        this.alphas[i] = 0;
        continue;
      }
      const speed = this.velocity.length();
      const s = THREE.MathUtils.clamp(
        (Math.log10(Math.max(speed, 1)) - LOG_SPEED_RANGE.x) / (LOG_SPEED_RANGE.y - LOG_SPEED_RANGE.x),
        0.05,
        1,
      );
      const ux = speed > 0 ? this.velocity.x / speed : 0;
      const uy = speed > 0 ? this.velocity.y / speed : 0;
      const step = s * r * 0.05 * dt;
      this.x[i] += ux * step;
      this.y[i] += uy * step;
      const length = r * 0.02 * (0.3 + 0.7 * s);
      this.segments.set([this.x[i] - ux * length, this.y[i] - uy * length, this.x[i], this.y[i]], i * 4);
      // Fade in and out
      const t = this.age[i] / this.life[i];
      this.alphas[i] = 0.85 * Math.min(1, t * 5, (1 - t) * 5);
    }
    const geometry = this.mesh.geometry;
    geometry.getAttribute("seg").needsUpdate = true;
    geometry.getAttribute("alpha").needsUpdate = true;
  }
}
