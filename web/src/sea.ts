/**
 * The sea surface: a flat layer at 0 m, the height of the geoid to which the
 * Bedmap3 elevations are referred (the mean sea level, up to the mean dynamic
 * topography and the tides, too small to show), over the sea: where Bedmap3
 * has no ice surface and the bed is below 0 m, within the data of the grid. The cells of the coarse
 * grid around the sea are drawn, then cut per pixel along the ice surface and
 * the coast with the height textures, so it follows the same shape as the
 * surface layer. Half transparent, darker over deeper water, with moving waves.
 */
import * as THREE from "three";
import { gridGeometry, gridPositions, type Grid } from "./terrain";

/** Heights (m) of a layer on a grid, and where it has data. */
interface Layer {
  heights: ArrayLike<number>;
  valid: ArrayLike<number>;
}

/** The height texture of a layer and its grid: the uniforms of its material, shared. */
export interface HeightUniforms {
  heightTex: THREE.IUniform;
  texOrigin: THREE.IUniform;
  texDx: THREE.IUniform;
  texSize: THREE.IUniform;
}

const vertexShader = /* glsl */ `
varying vec2 vPos; // km
void main() {
  vPos = position.xy;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = /* glsl */ `
uniform sampler2D bedTex;
uniform vec2 bedOrigin; // centre of the first texel, km
uniform float bedDx; // km
uniform vec2 bedSize; // texels
uniform sampler2D surfaceTex;
uniform vec2 surfaceOrigin;
uniform float surfaceDx;
uniform vec2 surfaceSize;
uniform float time; // s
uniform float opacity;
uniform vec2 clearAt; // km, centre of a patch of clearer water (over the wreck of the Endurance)
uniform float clearRadius; // km, 0 for none
varying vec2 vPos;

vec4 heightSample(sampler2D tex, vec2 origin, float dx, vec2 size) {
  return texture2D(tex, (vPos - origin) / (dx * size) + 0.5 / size);
}

// Height of the waves (no unit): a few trains of different lengths (km) and directions
float waves(vec2 p) {
  float h = 0.0;
  h += sin(dot(p, vec2(0.8, 0.6)) * 1.3 - time * 1.1);
  h += 0.7 * sin(dot(p, vec2(-0.5, 0.86)) * 2.1 - time * 1.5);
  h += 0.5 * sin(dot(p, vec2(0.95, -0.3)) * 3.7 - time * 2.0);
  h += 0.35 * sin(dot(p, vec2(-0.2, -0.98)) * 6.1 - time * 2.6);
  return h;
}

void main() {
  // Not over the ice (where the surface layer is drawn), nor over land
  if (heightSample(surfaceTex, surfaceOrigin, surfaceDx, surfaceSize).g >= 0.5) discard;
  vec4 bed = heightSample(bedTex, bedOrigin, bedDx, bedSize);
  // Nor out of the Bedmap3 grid (the round edge of its data)
  if (bed.g < 0.5 || bed.r >= 0.0) discard;

  // Lighter over the shelf seas, darker over the deep ocean
  float depth = clamp(-bed.r / 4000.0, 0.0, 1.0);
  vec3 water = mix(vec3(0.36, 0.66, 0.84), vec3(0.05, 0.2, 0.42), sqrt(depth));

  // Slopes of the waves, faded out where they are smaller than a pixel (no flicker from afar)
  float e = 0.05;
  vec2 slope = vec2(waves(vPos + vec2(e, 0.0)) - waves(vPos - vec2(e, 0.0)),
                    waves(vPos + vec2(0.0, e)) - waves(vPos - vec2(0.0, e))) / (2.0 * e);
  float fade = 1.0 - smoothstep(0.15, 0.6, length(fwidth(vPos)));
  // Clearer and calmer over the patch, to see down to the bed
  float clear = clearRadius > 0.0 ? 1.0 - smoothstep(0.6 * clearRadius, clearRadius, distance(vPos, clearAt)) : 0.0;
  fade *= 1.0 - 0.8 * clear;
  vec3 n = normalize(vec3(-0.2 * fade * slope, 1.0));
  vec3 light = normalize(vec3(-1.0, 1.0, 1.0));
  float diffuse = 0.7 + 0.3 * dot(n, light);
  // Light on the crests
  float glint = pow(max(dot(reflect(-light, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0) * fade;
  gl_FragColor = vec4(water * diffuse + 0.6 * glint, mix(opacity, 0.15, clear) + 0.3 * glint);
}
`;

export class Sea {
  readonly mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>;

  /** bed and surface: the height uniforms of the terrain layers, followed as they are refined. */
  constructor(bed: HeightUniforms, surface: HeightUniforms) {
    this.mesh = new THREE.Mesh(
      new THREE.BufferGeometry(),
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        uniforms: {
          bedTex: bed.heightTex,
          bedOrigin: bed.texOrigin,
          bedDx: bed.texDx,
          bedSize: bed.texSize,
          surfaceTex: surface.heightTex,
          surfaceOrigin: surface.texOrigin,
          surfaceDx: surface.texDx,
          surfaceSize: surface.texSize,
          time: { value: 0 },
          opacity: { value: 0.6 },
          clearAt: { value: new THREE.Vector2() },
          clearRadius: { value: 0 },
        },
      }),
    );
    this.mesh.frustumCulled = false;
  }

  /** The cells of the grid next to the sea, from the bed and the ice surface (at z = 0 already). */
  build(grid: Grid, bed: Layer, surface: Layer): void {
    const sea = new Uint8Array(grid.nx * grid.ny);
    for (let i = 0; i < sea.length; i++) sea[i] = !surface.valid[i] && bed.valid[i] && bed.heights[i] < 0 ? 1 : 0;
    this.mesh.geometry.dispose();
    this.mesh.geometry = gridGeometry(grid, gridPositions(grid), sea);
  }

  /** A patch of clearer, calmer water around a point (km), radius in km (0 for none). */
  setClearing(x: number, y: number, radius: number): void {
    this.mesh.material.uniforms.clearAt.value.set(x, y);
    this.mesh.material.uniforms.clearRadius.value = radius;
  }

  /** Move the waves on to a time (s). */
  setTime(time: number): void {
    this.mesh.material.uniforms.time.value = time;
  }
}
