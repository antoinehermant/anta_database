/**
 * Loading of the Bedmap3 heightmaps exported by anta_database.web.terrain,
 * and the meshes that display them.
 *
 * Scene units are km in EPSG:3031, z points up. Heights are applied in the
 * vertex shader from a texture, so the vertical exaggeration is a uniform and
 * the mesh can be coarser than the texture used for shading.
 */
import * as THREE from "three";
import vertexShader from "./shaders/terrain.vert.glsl?raw";
import fragmentShader from "./shaders/terrain.frag.glsl?raw";

export interface LayerInfo {
  file: string;
  min?: number;
  max?: number;
  dtype?: string;
}

export interface Level {
  nx: number;
  ny: number;
  dx: number; // m
  x0: number; // m, centre of the first cell
  y0: number;
  layers: Record<string, LayerInfo>;
}

export interface TerrainMeta {
  nodata: number;
  citation: string;
  doi: string;
  resolutions: Record<string, Level>;
}

/** Grid of a texture or mesh, in km. */
export interface Grid {
  nx: number;
  ny: number;
  dx: number;
  x0: number;
  y0: number;
}

export function gridOf(level: Level): Grid {
  return { nx: level.nx, ny: level.ny, dx: level.dx / 1000, x0: level.x0 / 1000, y0: level.y0 / 1000 };
}

export async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer, 0, 2);
  // Servers may already have decompressed it (Content-Encoding: gzip)
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return buffer;
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

export async function loadMeta(baseUrl: string): Promise<TerrainMeta> {
  const response = await fetch(`${baseUrl}/terrain.json`);
  if (!response.ok) {
    throw new Error(
      `${baseUrl}/terrain.json not found. Export it with: python -m anta_database.web.terrain bedmap3.nc -o web/public/data/terrain`,
    );
  }
  return response.json();
}

/**
 * Give nodata cells next to data the mean of their valid neighbours, a few
 * cells deep. They stay hidden, but triangles straddling the edge of the data
 * then don't plunge to 0 m and draw walls.
 */
function fillEdges(heights: Int16Array, valid: Uint8Array, nx: number, ny: number, passes: number): Float32Array {
  const out = Float32Array.from(heights);
  let known = valid.slice();
  for (let pass = 0; pass < passes; pass++) {
    const next = known.slice();
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (known[k]) continue;
        let sum = 0;
        let n = 0;
        if (i > 0 && known[k - 1]) (sum += out[k - 1]), n++;
        if (i < nx - 1 && known[k + 1]) (sum += out[k + 1]), n++;
        if (j > 0 && known[k - nx]) (sum += out[k - nx]), n++;
        if (j < ny - 1 && known[k + nx]) (sum += out[k + nx]), n++;
        if (n) {
          out[k] = sum / n;
          next[k] = 1;
        }
      }
    }
    known = next;
  }
  for (let k = 0; k < out.length; k++) if (!known[k]) out[k] = 0;
  return out;
}

/**
 * Height texture: R = elevation (m), G = 1 where there is data, 0 elsewhere.
 * Half floats are linearly filterable on every WebGL2 device.
 *
 * edgeCells: how far (in texels) heights are extended past the edge of the
 * data; at least the mesh/texture resolution ratio.
 */
export async function loadHeightTexture(
  baseUrl: string,
  level: Level,
  layer: string,
  nodata: number,
  edgeCells = 4,
): Promise<THREE.DataTexture> {
  const raw = new Int16Array(await fetchBytes(`${baseUrl}/${level.layers[layer].file}`));
  const valid = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) valid[i] = raw[i] === nodata ? 0 : 1;
  const heights = fillEdges(raw, valid, level.nx, level.ny, edgeCells);

  const data = new Uint16Array(heights.length * 2);
  const one = THREE.DataUtils.toHalfFloat(1);
  const halves = new Map<number, number>();
  for (let i = 0; i < heights.length; i++) {
    const h = Math.round(heights[i]);
    if (valid[i]) data[2 * i + 1] = one;
    if (h === 0) continue;
    let half = halves.get(h);
    if (half === undefined) {
      half = THREE.DataUtils.toHalfFloat(h);
      halves.set(h, half);
    }
    data[2 * i] = half;
    data[2 * i + 1] = one;
  }
  const texture = new THREE.DataTexture(data, level.nx, level.ny, THREE.RGFormat, THREE.HalfFloatType);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

/** Bedmap3 mask classes (0 ocean ... 4 rock), nearest filtered. */
export async function loadMaskTexture(baseUrl: string, level: Level): Promise<THREE.DataTexture> {
  const mask = new Uint8Array(await fetchBytes(`${baseUrl}/${level.layers.mask.file}`));
  const texture = new THREE.DataTexture(mask, level.nx, level.ny, THREE.RedFormat, THREE.UnsignedByteType);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}

/** Flat regular grid of vertices covering a grid, shared by all layers. */
export function gridGeometry(grid: Grid): THREE.BufferGeometry {
  const { nx, ny, dx, x0, y0 } = grid;
  const positions = new Float32Array(nx * ny * 3);
  for (let j = 0, k = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++, k += 3) {
      positions[k] = x0 + i * dx;
      positions[k + 1] = y0 + j * dx;
    }
  }
  const indices = new Uint32Array((nx - 1) * (ny - 1) * 6);
  for (let j = 0, k = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++, k += 6) {
      const a = j * nx + i;
      const b = a + 1;
      const c = a + nx;
      const d = c + 1;
      indices.set([a, b, d, a, d, c], k);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  return geometry;
}

/** 1D colormap texture from [elevation (m), css colour] stops. */
export function colormapTexture(stops: [number, string][], size = 512): {
  texture: THREE.DataTexture;
  range: THREE.Vector2;
} {
  const min = stops[0][0];
  const max = stops[stops.length - 1][0];
  const data = new Uint8Array(size * 4);
  const color = new THREE.Color();
  const a = new THREE.Color();
  const b = new THREE.Color();
  for (let i = 0; i < size; i++) {
    const h = min + ((max - min) * i) / (size - 1);
    let s = 0;
    while (s < stops.length - 2 && h > stops[s + 1][0]) s++;
    const t = THREE.MathUtils.clamp((h - stops[s][0]) / (stops[s + 1][0] - stops[s][0]), 0, 1);
    color.lerpColors(a.set(stops[s][1]), b.set(stops[s + 1][1]), t);
    data.set([color.r * 255, color.g * 255, color.b * 255, 255], i * 4);
  }
  const texture = new THREE.DataTexture(data, size, 1, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return { texture, range: new THREE.Vector2(min, max) };
}

export interface TerrainUniforms {
  [name: string]: THREE.IUniform;
}

/** Uniforms shared by all layers (exaggeration, light). */
export function sharedUniforms(): TerrainUniforms {
  return {
    exaggeration: { value: 30 },
    lightDir: { value: new THREE.Vector3(-1, 1, 1).normalize() },
    ambient: { value: 0.35 },
  };
}

export interface LayerOptions {
  colormap?: { texture: THREE.DataTexture; range: THREE.Vector2 };
  // Constant colour when there is no colormap
  color?: THREE.Color;
  // Colour of floating ice shelves, using the mask texture
  shelfColor?: THREE.Color;
  opacity?: number;
}

export function terrainMaterial(shared: TerrainUniforms, options: LayerOptions): THREE.ShaderMaterial {
  const transparent = (options.opacity ?? 1) < 1;
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    transparent,
    depthWrite: !transparent,
    side: THREE.DoubleSide,
    defines: {
      USE_COLORMAP: options.colormap ? 1 : 0,
    },
    uniforms: {
      ...shared,
      heightTex: { value: null },
      maskTex: { value: null },
      // x0, y0, dx (km) and size (texels) of the height texture
      texOrigin: { value: new THREE.Vector2() },
      texDx: { value: 1 },
      texSize: { value: new THREE.Vector2(1, 1) },
      colormapTex: { value: options.colormap?.texture ?? null },
      colormapRange: { value: options.colormap?.range ?? new THREE.Vector2(0, 1) },
      baseColor: { value: options.color ?? new THREE.Color("#ffffff") },
      shelfColor: { value: options.shelfColor ?? options.color ?? new THREE.Color("#ffffff") },
      opacity: { value: options.opacity ?? 1 },
    },
  });
}

/** Point a layer material at a (new) height texture and mask. */
export function setLayerTextures(
  material: THREE.ShaderMaterial,
  height: THREE.DataTexture,
  mask: THREE.DataTexture,
  grid: Grid,
): void {
  const u = material.uniforms;
  u.heightTex.value?.dispose();
  u.heightTex.value = height;
  u.maskTex.value = mask;
  u.texOrigin.value.set(grid.x0, grid.y0);
  u.texDx.value = grid.dx;
  u.texSize.value.set(grid.nx, grid.ny);
}
