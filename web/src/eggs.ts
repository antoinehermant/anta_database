/**
 * Hidden things to find: an alien vessel crashed into the Transantarctic
 * Mountains, near Mount Kirkpatrick; the fossil of Charybdis, the sea monster
 * of Greek myth, in a deep trough under the ice, towards the coast from the
 * lines of Bodart and Sutter (2025b); and the whirlpool she left in the
 * Southern Ocean, off the Wilkes Subglacial Basin.
 *
 * Unlike the aircraft, they have a fixed size in the scene (km), so they are
 * tiny from afar and grow as you zoom in on them. Their shapes aren't
 * exaggerated like the terrain, except the whirlpool, which spans the depth of
 * the ocean. The fossil lies on the bed as it is drawn: its vertices are moved
 * onto the triangles of the bed mesh in the vertex shader.
 */
import * as THREE from "three";
import { polarStereographic } from "./icecores";

/** Shading without lights: a sphere lit from the top left, as a matcap texture. */
function matcap(light: string, dark: string, highlight = "#ffffff"): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const context = canvas.getContext("2d")!;
  const gradient = context.createRadialGradient(44, 40, 4, 64, 64, 70);
  gradient.addColorStop(0, highlight);
  gradient.addColorStop(0.25, light);
  gradient.addColorStop(1, dark);
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function mesh(geometry: THREE.BufferGeometry, material: THREE.Material): THREE.Mesh {
  const m = new THREE.Mesh(geometry, material);
  m.frustumCulled = false;
  return m;
}

/** A flying saucer, 1 unit across, its axis along z. */
function vessel(): THREE.Group {
  const group = new THREE.Group();
  const hull = new THREE.MeshMatcapMaterial({ matcap: matcap("#b9c2cc", "#3c434c") });
  // Profile of the hull, turned around the axis
  const profile = [
    [0, -0.12],
    [0.22, -0.11],
    [0.5, -0.01],
    [0.5, 0.01],
    [0.3, 0.07],
    [0, 0.08],
  ].map(([r, z]) => new THREE.Vector2(r, z));
  group.add(mesh(new THREE.LatheGeometry(profile, 48).rotateX(Math.PI / 2), hull));
  // Glass dome, and the ring of lights, still on
  const dome = new THREE.MeshMatcapMaterial({ matcap: matcap("#9ff5c8", "#1d5a45"), transparent: true, opacity: 0.85 });
  group.add(mesh(new THREE.SphereGeometry(0.16, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2).rotateX(Math.PI / 2).translate(0, 0, 0.06), dome));
  const light = new THREE.MeshBasicMaterial({ color: "#7dfcff" });
  const bulb = new THREE.SphereGeometry(0.025, 8, 6);
  for (let k = 0; k < 12; k++) {
    const a = (k / 12) * Math.PI * 2;
    const m = mesh(bulb, light);
    m.position.set(0.42 * Math.cos(a), 0.42 * Math.sin(a), 0.03);
    group.add(m);
  }
  return group;
}

/** A curve through points (x, y, z rows). */
function curve(points: number[][]): THREE.CatmullRomCurve3 {
  return new THREE.CatmullRomCurve3(points.map(([x, y, z]) => new THREE.Vector3(x, y, z)));
}

/**
 * The fossil of Charybdis, 1 unit across, lying flat (relief along +z): one
 * big head, a round mouth full of teeth opening up, and many curling
 * tentacles. Dark, like a crocodile, so it is hard to see on the bed.
 */
function fossil(): THREE.Group {
  const group = new THREE.Group();
  const stone = new THREE.MeshMatcapMaterial({ matcap: matcap("#323a29", "#050604", "#57604a") });
  const bone = new THREE.MeshMatcapMaterial({ matcap: matcap("#7c775f", "#1f1d16", "#a39d84") });
  const shadow = new THREE.MeshMatcapMaterial({ matcap: matcap("#1c2017", "#030302", "#2c3125") });
  const lips = new THREE.MeshMatcapMaterial({ matcap: matcap("#3a2f24", "#070504", "#54473a") });
  // Same shapes each time
  let seed = 7;
  const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;

  // The head, with the mouth on top: lips, throat and two rings of teeth leaning in
  const head = mesh(new THREE.SphereGeometry(1, 32, 16), stone);
  head.scale.set(0.17, 0.17, 0.07);
  group.add(head);
  const mouth = new THREE.Group();
  mouth.position.z = 0.06;
  mouth.add(mesh(new THREE.TorusGeometry(0.1, 0.022, 10, 40), lips));
  mouth.add(mesh(new THREE.CircleGeometry(0.1, 40).translate(0, 0, -0.005), new THREE.MeshBasicMaterial({ color: "#030302" })));
  for (const [ring, count, size] of [
    [0.09, 20, 0.035],
    [0.06, 14, 0.026],
  ]) {
    const tooth = new THREE.ConeGeometry(size * 0.3, size, 6).rotateX(Math.PI / 2).translate(0, 0, size / 2);
    for (let k = 0; k < count; k++) {
      const a = (k / count) * Math.PI * 2 + ring * 10;
      const m = mesh(tooth, bone);
      m.position.set(ring * Math.cos(a), ring * Math.sin(a), 0);
      // Up and towards the middle
      m.lookAt(new THREE.Vector3(-0.6 * Math.cos(a), -0.6 * Math.sin(a), 1).add(m.position));
      mouth.add(m);
    }
  }
  group.add(mouth);

  // Tentacles: out from the body, bending gently, then rolled up at the tip
  const TENTACLES = 14;
  for (let k = 0; k < TENTACLES; k++) {
    const a0 = (k / TENTACLES) * Math.PI * 2 + 0.2 * random();
    const length = 0.22 + 0.12 * random();
    const side = random() < 0.5 ? -1 : 1;
    const bend = side * (0.3 + 0.3 * random());
    const points: number[][] = [];
    const wave = (t: number) => 0.004 * Math.sin(t * 9 + k);
    for (let i = 0; i <= 16; i++) {
      const t = i / 16;
      const a = a0 + bend * t * t;
      const r = 0.1 + length * t;
      points.push([r * Math.cos(a), r * Math.sin(a), wave(t)]);
    }
    // The curl: a shrinking spiral around a point beside the end, turning to the same side as the bend
    const [ex, ey] = points[points.length - 1];
    const [px, py] = points[points.length - 2];
    const heading = Math.atan2(ey - py, ex - px);
    const rc = 0.045 + 0.02 * random();
    const cx = ex + rc * Math.cos(heading + side * Math.PI / 2);
    const cy = ey + rc * Math.sin(heading + side * Math.PI / 2);
    const start = heading - side * Math.PI / 2;
    for (let i = 1; i <= 14; i++) {
      const u = i / 14;
      const phi = start + side * u * Math.PI * 1.7;
      const r = rc * (1 - 0.65 * u);
      points.push([cx + r * Math.cos(phi), cy + r * Math.sin(phi), wave(1)]);
    }
    const radius = 0.022 - 0.006 * random();
    const path = curve(points);
    group.add(mesh(new THREE.TubeGeometry(path, 64, radius, 8), stone));
    // Suckers along the tentacle, on top of it
    const sucker = new THREE.SphereGeometry(radius * 0.45, 6, 4);
    for (let i = 2; i < 12; i++) {
      const m = mesh(sucker, shadow);
      const p = path.getPoint(i / 16);
      m.position.set(p.x, p.y, p.z + radius * 0.75);
      group.add(m);
    }
  }

  return group;
}

interface Egg {
  title: string;
  rows: [string, string][];
  note: string;
  objects: THREE.Object3D[];
}

/** Uniforms of the bed: its height texture and grid, the mesh grid, and the exaggeration. */
export interface BedUniforms {
  heightTex: THREE.IUniform;
  texOrigin: THREE.IUniform;
  texDx: THREE.IUniform;
  texSize: THREE.IUniform;
  exaggeration: THREE.IUniform;
  meshOrigin: THREE.IUniform<THREE.Vector2>;
  meshDx: THREE.IUniform<number>;
}

/**
 * Put the vertices of a material on the bed mesh: the height of the bed under
 * each vertex (in the world), on the same triangles as the mesh, is added to
 * its z, so the object lies on the bed whatever its relief.
 */
function onBed(material: THREE.Material, bed: BedUniforms): void {
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, bed);
    shader.vertexShader = shader.vertexShader
      .replace(
        "void main() {",
        `uniform sampler2D heightTex;
uniform vec2 texOrigin;
uniform float texDx;
uniform vec2 texSize;
uniform float exaggeration;
uniform vec2 meshOrigin;
uniform float meshDx;
float bedAt(vec2 p) {
  return texture2D(heightTex, (p - texOrigin) / (texDx * texSize) + 0.5 / texSize).r;
}
void main() {`,
      )
      .replace(
        "#include <project_vertex>",
        `vec4 bedWorld = modelMatrix * vec4(transformed, 1.0);
vec2 cell = (bedWorld.xy - meshOrigin) / meshDx;
vec2 i = floor(cell);
vec2 f = cell - i;
vec2 a = meshOrigin + i * meshDx;
float ha = bedAt(a);
float hb = bedAt(a + vec2(meshDx, 0.0));
float hc = bedAt(a + vec2(0.0, meshDx));
float hd = bedAt(a + vec2(meshDx));
// Triangles (a, b, d) and (a, d, c) of the mesh
float h = f.x > f.y ? ha + f.x * (hb - ha) + f.y * (hd - hb) : ha + f.y * (hc - ha) + f.x * (hd - hc);
bedWorld.z += h * exaggeration / 1000.0;
vec4 mvPosition = viewMatrix * bedWorld;
gl_Position = projectionMatrix * mvPosition;`,
      );
  };
}

/** Faint stripes winding around the whirlpool: a texture repeated around the funnel. */
function spiralTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 256;
  const context = canvas.getContext("2d")!;
  context.fillStyle = "rgba(40, 110, 190, 0.03)";
  context.fillRect(0, 0, 256, 256);
  context.strokeStyle = "rgba(120, 190, 255, 0.1)";
  context.lineWidth = 10;
  // Diagonal lines become spirals around the funnel
  for (let k = -256; k < 512; k += 64) {
    context.beginPath();
    context.moveTo(k, 256);
    context.lineTo(k + 256, 0);
    context.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 3);
  return texture;
}

// Sizes (km) and places
const VESSEL_SIZE = 9;
const VESSEL_AT = polarStereographic(-84.36, 165.6); // a flank of Mount Kirkpatrick
const FOSSIL_SIZE = 10;
const FOSSIL_RELIEF = 1.5; // relief made a bit higher, not to vanish in the bed
const FOSSIL_AT = polarStereographic(-69.72, 152.79); // a deep trough, coastward of Bodart and Sutter (2025b)
// Sunk into the bed (in units of its size): about half the thickness of the tentacles
const FOSSIL_BURIED = 0.01;
const WHIRLPOOL_RADIUS = 35; // at the sea surface, down to a point at the seafloor
const WHIRLPOOL_AT = polarStereographic(-64, 150); // off the Wilkes Subglacial Basin

export class EasterEggs {
  readonly group = new THREE.Group();
  private exaggeration = 1;
  private vessel = vessel();
  private fossil = fossil();
  private whirlpool: THREE.Mesh<THREE.CylinderGeometry, THREE.MeshBasicMaterial>;
  private eggs: Egg[];
  private fossilRows: [string, string][];
  private raycaster = new THREE.Raycaster();

  /**
   * heightAt(x, y): elevation (m) of the top of the ground or ice, bedAt(x, y)
   * and surfaceAt(x, y): of the bed and of the ice surface, null without data;
   * bed: the uniforms of the bed, to lie on it.
   */
  constructor(
    private heightAt: (x: number, y: number) => number | null,
    private bedAt: (x: number, y: number) => number | null,
    private surfaceAt: (x: number, y: number) => number | null,
    private bed: BedUniforms,
  ) {
    this.fossil.traverse((object) => {
      if (object instanceof THREE.Mesh) onBed(object.material, bed);
    });
    // Open cone, 1 unit high, from its point at z = 0 (the seafloor) to the sea surface
    const cone = new THREE.CylinderGeometry(1, 0.01, 1, 64, 12, true).rotateX(Math.PI / 2).translate(0, 0, 0.5);
    this.whirlpool = new THREE.Mesh(
      cone,
      new THREE.MeshBasicMaterial({ map: spiralTexture(), transparent: true, depthWrite: false, side: THREE.DoubleSide }),
    );
    this.whirlpool.frustumCulled = false;
    this.vessel.scale.setScalar(VESSEL_SIZE);
    // Nose down into the slope
    this.vessel.rotation.set(0.75, -0.35, 0.4);
    this.fossil.scale.set(FOSSIL_SIZE, FOSSIL_SIZE, FOSSIL_SIZE * FOSSIL_RELIEF);
    this.fossil.rotation.z = 2.2;
    this.group.add(this.vessel, this.fossil, this.whirlpool);
    this.fossilRows = [
      ["Position", "69.7° S, 152.8° E, at the bottom of a deep trough"],
      ["Near", "the lines of Bodart and Sutter (2025b)"],
      ["Age", "older than the oldest isochrone"],
      ["Searched by", "Johannes Sutter, still on the quest to find it"],
    ];
    this.eggs = [
      {
        title: "Unidentified crashed vessel",
        rows: [
          ["Position", "84.4° S, 165.6° E, Mount Kirkpatrick"],
          ["Origin", "unknown"],
          ["Crew", "not found"],
        ],
        note: "Not in Bedmap3. Lights still on.",
        objects: [this.vessel],
      },
      {
        title: "Fossil of Charybdis",
        rows: this.fossilRows,
        note: "The sea monster of Greek myth (Homer, Odyssey XII), frozen in the basal ice. You found it before Johannes did.",
        objects: [this.fossil],
      },
      {
        title: "Remnant whirlpool of Charybdis",
        rows: [
          ["Position", "64° S, 150° E, off the Wilkes Subglacial Basin"],
          ["Depth", "all the way down to the seafloor, about 3.6 km"],
          ["Activity", "swallows the sea three times a day, and spits it out again"],
          ["Last seen", "by Odysseus, who survived by clinging to a fig tree"],
        ],
        note: "Left over from the activity of Charybdis, before she settled inland under the ice. Still spinning.",
        objects: [this.whirlpool],
      },
    ];
    this.place();
  }

  setExaggeration(exaggeration: number): void {
    this.exaggeration = exaggeration;
    this.place();
  }

  /** Put them at the heights of the terrain, once it is loaded (and again with the exaggeration). */
  place(): void {
    const z = (m: number) => (m * this.exaggeration) / 1000;
    const [vx, vy] = VESSEL_AT;
    const ground = this.heightAt(vx, vy);
    this.vessel.visible = ground !== null;
    // Half buried
    if (ground !== null) this.vessel.position.set(vx, vy, z(ground) - 0.1 * VESSEL_SIZE);
    // On the bed (moved onto it in the shader), partly sunk into it
    const [fx, fy] = FOSSIL_AT;
    const floor = this.bedAt(fx, fy);
    this.fossil.visible = floor !== null;
    this.fossil.position.set(fx, fy, -FOSSIL_BURIED * FOSSIL_SIZE * FOSSIL_RELIEF);
    const surface = this.surfaceAt(fx, fy);
    const ice = surface === null || floor === null ? null : Math.round((surface - floor) / 10) * 10;
    this.fossilRows.splice(4, 1, ...(ice === null ? [] : [["Ice above", `about ${ice} m`] as [string, string]]));

    // From the sea surface down to the seafloor
    const [wx, wy] = WHIRLPOOL_AT;
    const depth = this.bedAt(wx, wy);
    this.whirlpool.visible = depth !== null && depth < 0;
    if (depth === null || depth >= 0) return;
    this.whirlpool.position.set(wx, wy, z(depth));
    this.whirlpool.scale.set(WHIRLPOOL_RADIUS, WHIRLPOOL_RADIUS, -z(depth));
  }

  /** Turn the whirlpool, by dt seconds. */
  update(dt: number): void {
    const map = this.whirlpool.material.map!;
    map.offset.x = (map.offset.x + 0.08 * dt) % 1;
    map.offset.y = (map.offset.y - 0.04 * dt) % 1;
  }

  /** The thing under a point of the canvas (px), or null. */
  pick(camera: THREE.Camera, x: number, y: number, width: number, height: number): Egg | null {
    if (!this.group.visible) return null;
    this.raycaster.setFromCamera(new THREE.Vector2((x / width) * 2 - 1, 1 - (y / height) * 2), camera);
    for (const egg of this.eggs) {
      if (egg.objects[0] === this.fossil) {
        if (this.overFossil(camera, x, y, width, height)) return egg;
      } else if (this.raycaster.intersectObjects(egg.objects, true).length) return egg;
    }
    return null;
  }

  /**
   * Whether a pixel is over the fossil, which the raycaster can't see (it is
   * moved onto the bed on the GPU): within its radius on screen around its
   * centre on the bed.
   */
  private overFossil(camera: THREE.Camera, x: number, y: number, width: number, height: number): boolean {
    if (!this.fossil.visible) return false;
    const [fx, fy] = FOSSIL_AT;
    const z = ((this.bedAt(fx, fy) ?? 0) * this.exaggeration) / 1000;
    const centre = new THREE.Vector3(fx, fy, z).project(camera);
    const edge = new THREE.Vector3(fx + 0.4 * FOSSIL_SIZE, fy, z).project(camera);
    if (centre.z > 1) return false; // behind the camera
    const px = (v: THREE.Vector3) => [((v.x + 1) / 2) * width, ((1 - v.y) / 2) * height];
    const [cx, cy] = px(centre);
    const [ex, ey] = px(edge);
    return Math.hypot(x - cx, y - cy) <= Math.max(6, Math.hypot(ex - cx, ey - cy));
  }
}
