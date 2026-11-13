/**
 * Hidden things to find: an alien vessel crashed into the Transantarctic
 * Mountains, near Mount Kirkpatrick; the fossil of Charybdis, the sea monster
 * of Greek myth, in a deep trough under the ice, towards the coast from the
 * lines of Bodart and Sutter (2025b); and the whirlpool she left in the
 * Southern Ocean, off the Wilkes Subglacial Basin; and the cromlech of ice
 * cores standing on the surface next to Little Dome C; and the glass of whisky
 * of Claude Lorius, on a table outside Dumont d'Urville station; and a train
 * like Snowpiercer, going round and round Skytrain Ice Rise; and the yeti of
 * Yetisports on an island in Pine Island Bay, playing Orca Slap with the
 * penguins, aiming at a target on an iceberg.
 *
 * Unlike the aircraft, they have a fixed size in the scene (km), so they are
 * tiny from afar and grow as you zoom in on them. Their shapes aren't
 * exaggerated like the terrain, except the whirlpool, which spans the depth of
 * the ocean. The fossil lies on the bed as it is drawn: its vertices are moved
 * onto the triangles of the bed mesh in the vertex shader.
 */
import * as THREE from "three";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
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

/**
 * A ring of ice cores standing on end, 1 unit across, on z = 0: like the
 * circle of about 50 one-metre cores that a team of women scientists set up
 * at Little Dome C during the 2023–2024 drilling season. Each core is a bit
 * different in height and leans a little, as cut and stood by hand; some are
 * broken, their pieces lying on the snow around them.
 */
function cromlech(): THREE.Group {
  const group = new THREE.Group();
  const ice = new THREE.MeshMatcapMaterial({ matcap: matcap("#e4f1ff", "#6f8fb0") });
  let seed = 11;
  const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
  const CORES = 50;
  const radius = 0.012;
  const core = new THREE.CylinderGeometry(radius, radius, 1, 12).rotateX(Math.PI / 2).translate(0, 0, 0.5);
  // A piece lying down: along y, centred
  const piece = new THREE.CylinderGeometry(radius, radius, 1, 12);
  const HEIGHT = 0.12; // a whole core
  for (let k = 0; k < CORES; k++) {
    const a = (k / CORES) * Math.PI * 2 + 0.02 * (random() - 0.5);
    const [x, y] = [0.5 * Math.cos(a), 0.5 * Math.sin(a)];
    const height = HEIGHT * (0.92 + 0.16 * random());
    // About two in five are broken: a stump left standing, the rest in one to three pieces
    const broken = random() < 0.4;
    const standing = broken ? height * (0.25 + 0.45 * random()) : height;
    const m = mesh(core, ice);
    m.position.set(x, y, 0);
    m.scale.z = standing;
    // Standing: can be knocked over (EasterEggs.topple)
    m.userData.core = { radius, piece, ice };
    m.rotation.set(0.06 * (random() - 0.5), 0.06 * (random() - 0.5), 0);
    group.add(m);
    if (!broken) continue;
    const pieces = 1 + Math.floor(3 * random());
    let left = height - standing;
    for (let p = 0; p < pieces; p++) {
      const length = p === pieces - 1 ? left : left * (0.4 + 0.3 * random());
      left -= length;
      // Most kicked into the middle of the circle, some fallen just outside
      const out = random() < 0.3 ? 0.02 + 0.05 * random() : -(0.03 + 0.35 * random());
      const side = 0.08 * (random() - 0.5);
      const fallen = mesh(piece, ice);
      fallen.position.set(x + out * Math.cos(a) - side * Math.sin(a), y + out * Math.sin(a) + side * Math.cos(a), radius);
      fallen.scale.y = length;
      fallen.rotation.z = Math.PI * random();
      group.add(fallen);
    }
  }
  // Not drawn: the disc inside the ring, so the whole cromlech can be hovered
  const disc = new THREE.Mesh(new THREE.CircleGeometry(0.52, 32), new THREE.MeshBasicMaterial({ visible: false }));
  disc.position.z = 0.03;
  group.add(disc);
  return group;
}

/** A hut: walls of a colour and a pitched roof (ridge along x), w x d x h, on z = 0. */
function hut(w: number, d: number, h: number, walls: THREE.Material, roof: THREE.Material): THREE.Group {
  const group = new THREE.Group();
  const body = mesh(new THREE.BoxGeometry(w, d, h), walls);
  body.position.z = h / 2;
  group.add(body);
  // A prism: a cylinder of three sides, laid along x
  const top = mesh(new THREE.CylinderGeometry(d / Math.sqrt(3), d / Math.sqrt(3), w * 1.04, 3).rotateZ(Math.PI / 2).rotateX(-Math.PI / 2), roof);
  top.scale.z = 0.7;
  top.position.z = h + (0.7 * d) / (2 * Math.sqrt(3));
  group.add(top);
  return group;
}

/** Size of the flag (station units): along the wind from the pole, and high. */
const FLAG = [0.06, 0.04];

/**
 * The French flag, its edge on the pole (at the origin, along z): one sheet of
 * cells, to be waved in the wind (waveFlag). Shaded by its folds through the
 * vertex colours, as there is no light in the scene.
 */
function flag(): THREE.Mesh {
  const canvas = document.createElement("canvas");
  [canvas.width, canvas.height] = [48, 32];
  const context = canvas.getContext("2d")!;
  ["#1d3f8f", "#f4f4f4", "#d42a2a"].forEach((color, k) => {
    context.fillStyle = color;
    context.fillRect(16 * k, 0, 16, 32);
  });
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const [w, h] = FLAG;
  const geometry = new THREE.PlaneGeometry(w, h, 16, 6).translate(w / 2, 0, 0).rotateX(Math.PI / 2);
  geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(geometry.attributes.position.count * 3).fill(1), 3));
  geometry.userData.rest = (geometry.attributes.position.array as Float32Array).slice();
  const m = mesh(geometry, new THREE.MeshBasicMaterial({ map: texture, vertexColors: true, side: THREE.DoubleSide }));
  m.name = "flag";
  m.position.set(0.05, 0.2, 0.18);
  return m;
}

/**
 * The flag at a time (s): waves running from the pole to the free edge, growing along it, the
 * free edge drooping a little; darker in the hollows.
 */
function waveFlag(flag: THREE.Mesh, time: number): void {
  const position = flag.geometry.attributes.position;
  const color = flag.geometry.attributes.color;
  const rest = flag.geometry.userData.rest as Float32Array;
  const [w] = FLAG;
  for (let i = 0; i < position.count; i++) {
    const [x, z] = [rest[3 * i], rest[3 * i + 2]];
    const u = x / w; // 0 at the pole, 1 at the free edge
    const phase = (x / w) * 9 - time * 7 + z * 40;
    const y = 0.006 * u * Math.sin(phase) + 0.002 * u * Math.sin(phase * 2.3 + 1);
    position.setXYZ(i, x * (1 - 0.04 * u * (1 + Math.cos(phase))), y, z - 0.004 * u * u);
    const shade = 0.82 + 0.18 * Math.cos(phase);
    color.setXYZ(i, shade, shade, shade);
  }
  position.needsUpdate = true;
  color.needsUpdate = true;
}

/**
 * A small coastal station, 1 unit across, on z = 0, after Dumont d'Urville on
 * the Île des Pétrels: a few huts, a radio mast and a French flag. Not an egg
 * itself: the egg is on the table outside the biggest hut (whiskyTable).
 */
function station(): THREE.Group {
  const group = new THREE.Group();
  const roof = new THREE.MeshMatcapMaterial({ matcap: matcap("#5d6670", "#1e2328") });
  const huts: [number, number, number, number, number, string, string][] = [
    // x, y, w, d, h, light, dark
    [0, 0, 0.34, 0.16, 0.09, "#d1513e", "#5c1a12"],
    [-0.3, 0.22, 0.22, 0.13, 0.07, "#e0b552", "#6b4a12"],
    [0.28, 0.25, 0.2, 0.12, 0.07, "#7aa071", "#25401f"],
    [-0.12, -0.3, 0.26, 0.12, 0.06, "#e9eef2", "#7d8a96"],
    [0.32, -0.2, 0.14, 0.1, 0.06, "#d1513e", "#5c1a12"],
  ];
  for (const [x, y, w, d, h, light, dark] of huts) {
    const m = hut(w, d, h, new THREE.MeshMatcapMaterial({ matcap: matcap(light, dark) }), roof);
    m.position.set(x, y, 0);
    group.add(m);
  }
  // Radio mast, with its cross bars
  const steel = new THREE.MeshMatcapMaterial({ matcap: matcap("#c9ced4", "#4a5058") });
  const mast = mesh(new THREE.CylinderGeometry(0.004, 0.006, 0.45, 6).rotateX(Math.PI / 2).translate(0, 0, 0.225), steel);
  mast.position.set(-0.42, -0.05, 0);
  group.add(mast);
  for (const z of [0.2, 0.3, 0.4]) {
    const bar = mesh(new THREE.BoxGeometry(0.06, 0.003, 0.003), steel);
    bar.position.set(-0.42, -0.05, z);
    group.add(bar);
  }
  // The flag, blue, white and red
  const pole = mesh(new THREE.CylinderGeometry(0.002, 0.002, 0.2, 6).rotateX(Math.PI / 2).translate(0, 0, 0.1), steel);
  pole.position.set(0.05, 0.2, 0);
  group.add(pole);
  group.add(flag());
  return group;
}

/**
 * The island under the station, 1 unit across at the top (z = 0), going down
 * to z = -1 (scaled to the seafloor): the Île des Pétrels is too small for the
 * 500 m grid of Bedmap3, which has sea there. Bumpy, not round. Snowy: white
 * on top, for the yeti.
 */
function island(snowy = false): THREE.Mesh {
  // From the foot up, for the faces to look outwards
  const profile = [
    [0.85, -1],
    [0.7, -0.4],
    [0.62, -0.12],
    [0.55, -0.02],
    // Flat on top, where the huts and the table stand (the height is scaled to the seafloor)
    [0.45, 0],
    [0, 0],
  ].map(([r, z]) => new THREE.Vector2(r, z));
  const geometry = new THREE.LatheGeometry(profile, 40).rotateX(Math.PI / 2);
  const position = geometry.attributes.position;
  for (let i = 0; i < position.count; i++) {
    const a = Math.atan2(position.getY(i), position.getX(i));
    const f = 1 + 0.12 * Math.sin(3 * a + 1) + 0.07 * Math.sin(5 * a + 2);
    position.setXY(i, position.getX(i) * f, position.getY(i) * f);
  }
  geometry.computeVertexNormals();
  if (!snowy) return mesh(geometry, new THREE.MeshMatcapMaterial({ matcap: matcap("#9b9893", "#4d4a46", "#bdbab5") }));
  // Snow on top and over the rim, grey rock below (vertex colours, darkening a white matcap)
  const colors = new Float32Array(position.count * 3);
  for (let i = 0; i < position.count; i++) colors.set(position.getZ(i) >= -0.02 ? [1, 1, 1] : [0.56, 0.55, 0.53], 3 * i);
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return mesh(geometry, new THREE.MeshMatcapMaterial({ matcap: matcap("#f2f6fa", "#8d99a6"), vertexColors: true }));
}

/**
 * A long armoured train, after Snowpiercer, along +x in its own frame (1 unit =
 * 1 car length with the gap), each car its own group to be moved along the
 * track in update(). Rounded steel cars on two bogies, joined by gangways, a
 * dull red stripe and lit windows (dark at the tail, like in the film); the
 * engine in front, dark, with a long rounded nose, a windscreen, a plough blade
 * and two headlights.
 */
function train(cars: number): THREE.Group[] {
  const steel = new THREE.MeshMatcapMaterial({ matcap: matcap("#b4bac2", "#2f343b", "#e8ecf0") });
  const dark = new THREE.MeshMatcapMaterial({ matcap: matcap("#545a63", "#0f1114", "#9aa1aa") });
  const rubber = new THREE.MeshMatcapMaterial({ matcap: matcap("#2c2f33", "#060708", "#4a4e54") });
  const stripe = new THREE.MeshMatcapMaterial({ matcap: matcap("#9c3a2e", "#3a110c", "#d07a6c") });
  const glass = new THREE.MeshMatcapMaterial({ matcap: matcap("#3d5a75", "#0b1520", "#a9c6de") });
  const lit = new THREE.MeshBasicMaterial({ color: "#ffd98a" });
  const unlit = new THREE.MeshMatcapMaterial({ matcap: matcap("#2a3038", "#07090b") });
  const lamp = new THREE.MeshBasicMaterial({ color: "#fff6c8" });

  const body = new RoundedBoxGeometry(0.9, 0.28, 0.34, 4, 0.08);
  const roof = new RoundedBoxGeometry(0.72, 0.16, 0.05, 3, 0.02);
  const band = new RoundedBoxGeometry(0.9, 0.284, 0.025, 2, 0.01);
  const pane = new RoundedBoxGeometry(0.1, 0.286, 0.07, 3, 0.025);
  const bogie = new RoundedBoxGeometry(0.22, 0.22, 0.06, 3, 0.02);
  const wheel = new THREE.CylinderGeometry(0.035, 0.035, 0.24, 16);
  const gangway = new RoundedBoxGeometry(0.14, 0.2, 0.26, 3, 0.04);
  const part = (geometry: THREE.BufferGeometry, material: THREE.Material, x: number, y: number, z: number) => {
    const m = mesh(geometry, material);
    m.position.set(x, y, z);
    return m;
  };

  const groups: THREE.Group[] = [];
  for (let k = 0; k < cars; k++) {
    const car = new THREE.Group();
    const engine = k === 0;
    // Running gear: two bogies with their wheels
    for (const bx of [-0.3, 0.3]) {
      car.add(part(bogie, rubber, bx, 0, 0.04));
      for (const wx of [-0.065, 0.065]) car.add(part(wheel, rubber, bx + wx, 0, 0.035));
    }
    // To the car behind
    if (k < cars - 1) car.add(part(gangway, rubber, -0.5, 0, 0.21));
    if (engine) {
      // A shorter body, then the nose: a long rounded hood, sloping down to the plough
      car.add(part(new RoundedBoxGeometry(0.62, 0.28, 0.36, 4, 0.08), dark, -0.14, 0, 0.22));
      const nose = mesh(new THREE.SphereGeometry(0.15, 32, 20), dark);
      nose.position.set(0.18, 0, 0.2);
      nose.scale.set(2.1, 0.93, 1.05);
      car.add(nose);
      const windscreen = mesh(new THREE.SphereGeometry(0.15, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2.6), glass);
      windscreen.position.set(0.12, 0, 0.27);
      windscreen.scale.set(1.2, 0.8, 0.7);
      car.add(windscreen);
      const plough = mesh(new THREE.SphereGeometry(0.15, 32, 12), steel);
      plough.position.set(0.42, 0, 0.06);
      plough.scale.set(1.2, 1.05, 0.38);
      car.add(plough);
      for (const ly of [-0.07, 0.07]) car.add(part(new THREE.SphereGeometry(0.028, 12, 8), lamp, 0.47, ly, 0.17));
      car.add(part(band, stripe, -0.14, 0, 0.13));
      car.children[car.children.length - 1].scale.x = 0.62 / 0.9;
    } else {
      car.add(part(body, steel, 0, 0, 0.21));
      car.add(part(roof, steel, 0, 0, 0.385));
      car.add(part(band, stripe, 0, 0, 0.12));
      // Windows: lit, but dark in the last cars
      const tail = k > cars - 4;
      for (let i = 0; i < 5; i++) car.add(part(pane, tail ? unlit : lit, -0.32 + i * 0.16, 0, 0.25));
    }
    groups.push(car);
  }
  return groups;
}

/**
 * Puffs of snow (km, in the frame of the train): thrown up at a point, falling
 * back and fading. Drawn as round soft points, their size in km.
 */
class SnowBursts {
  readonly points: THREE.Points;
  private velocities: Float32Array;
  private life: Float32Array;
  private next = 0;
  private positions: THREE.BufferAttribute;
  private alpha: THREE.BufferAttribute;

  constructor(
    private count = 3000,
    private size = 1.2,
  ) {
    const geometry = new THREE.BufferGeometry();
    this.positions = new THREE.BufferAttribute(new Float32Array(count * 3), 3);
    this.alpha = new THREE.BufferAttribute(new Float32Array(count), 1);
    geometry.setAttribute("position", this.positions);
    geometry.setAttribute("alpha", this.alpha);
    this.velocities = new Float32Array(count * 3);
    this.life = new Float32Array(count);
    const material = new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      uniforms: { size: { value: size }, scale: { value: 1 } },
      vertexShader: /* glsl */ `
        attribute float alpha;
        uniform float size; // km
        uniform float scale; // px per km at 1 km
        varying float vAlpha;
        void main() {
          vAlpha = alpha;
          vec4 p = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = alpha > 0.0 ? size * scale / -p.z * (2.5 - 1.5 * alpha) : 0.0;
          gl_Position = projectionMatrix * p;
        }`,
      fragmentShader: /* glsl */ `
        varying float vAlpha;
        void main() {
          float r = length(gl_PointCoord - 0.5) * 2.0;
          if (r > 1.0) discard;
          gl_FragColor = vec4(vec3(0.97, 0.98, 1.0), vAlpha * (1.0 - r * r) * 0.9);
        }`,
    });
    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    // Points the same size in km at any distance: pixels per km at 1 km from the camera
    this.points.onBeforeRender = (renderer, _scene, camera) => {
      const fov = (camera as THREE.PerspectiveCamera).fov ?? 45;
      material.uniforms.scale.value = renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(fov) / 2));
    };
  }

  /** Throw n flakes from a point, up and away, mostly along a direction (unit vector, horizontal); slower with speed < 1. */
  emit(at: THREE.Vector3, along: THREE.Vector3, n: number, speed = 1): void {
    for (let k = 0; k < n; k++) {
      const i = this.next;
      this.next = (this.next + 1) % this.count;
      const a = Math.random() * Math.PI * 2;
      const spread = 0.5 + 3 * Math.random() ** 2;
      const forward = (Math.random() - 0.2) * 4;
      this.positions.setXYZ(i, at.x + 0.8 * (Math.random() - 0.5), at.y + 0.8 * (Math.random() - 0.5), at.z + 0.3 * Math.random());
      this.velocities.set(
        [along.x * forward + spread * Math.cos(a), along.y * forward + spread * Math.sin(a), 1.5 + 6 * Math.random() ** 1.5].map((v) => v * speed),
        i * 3,
      );
      // Not all at once: the cloud lasts a while, some flakes longer
      this.life[i] = 0.7 + 0.3 * Math.random();
    }
  }

  /** Move them on by dt seconds: gravity, air braking, fading. */
  update(dt: number): void {
    const p = this.positions.array as Float32Array;
    const v = this.velocities;
    for (let i = 0; i < this.count; i++) {
      if (this.life[i] <= 0) continue;
      this.life[i] = Math.max(0, this.life[i] - dt / 3);
      const drag = Math.exp(-1.2 * dt);
      v[3 * i] *= drag;
      v[3 * i + 1] *= drag;
      v[3 * i + 2] = v[3 * i + 2] * drag - 3 * dt;
      p[3 * i] += v[3 * i] * dt;
      p[3 * i + 1] += v[3 * i + 1] * dt;
      p[3 * i + 2] += v[3 * i + 2] * dt;
      this.alpha.setX(i, this.life[i]);
    }
    this.positions.needsUpdate = true;
    this.alpha.needsUpdate = true;
  }
}

/**
 * Blocks of ice thrown out (km, in the frame of the train): tumbling, falling
 * back onto a level, bouncing once or twice, then shrinking away.
 */
class IceDebris {
  readonly mesh: THREE.InstancedMesh;
  private state: { p: THREE.Vector3; v: THREE.Vector3; spin: THREE.Vector3; q: THREE.Euler; size: number; life: number; floor: number }[] = [];
  private next = 0;
  private matrix = new THREE.Matrix4();
  private quaternion = new THREE.Quaternion();

  constructor(private count = 160) {
    const ice = new THREE.MeshMatcapMaterial({ matcap: matcap("#eef7ff", "#7ea6c8", "#ffffff") });
    this.mesh = new THREE.InstancedMesh(new THREE.DodecahedronGeometry(0.5, 0), ice, count);
    this.mesh.frustumCulled = false;
    for (let i = 0; i < count; i++) {
      this.state.push({ p: new THREE.Vector3(), v: new THREE.Vector3(), spin: new THREE.Vector3(), q: new THREE.Euler(), size: 0, life: 0, floor: 0 });
      this.mesh.setMatrixAt(i, this.matrix.makeScale(0, 0, 0));
    }
  }

  /** Throw n blocks from a point, mostly along a direction (unit vector, horizontal); floor: the ground level. */
  emit(at: THREE.Vector3, along: THREE.Vector3, n: number, floor: number): void {
    for (let k = 0; k < n; k++) {
      const d = this.state[this.next];
      this.next = (this.next + 1) % this.count;
      const a = Math.random() * Math.PI * 2;
      const spread = 0.5 + 2.5 * Math.random();
      const forward = (Math.random() - 0.1) * 5;
      d.p.set(at.x, at.y, at.z + 0.3);
      d.v.set(along.x * forward + spread * Math.cos(a), along.y * forward + spread * Math.sin(a), 2 + 5 * Math.random());
      d.spin.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5).multiplyScalar(12);
      d.q.set(Math.random() * 6, Math.random() * 6, Math.random() * 6);
      // A few big, many small
      d.size = 0.12 + 0.6 * Math.random() ** 3;
      d.life = 4 + 2 * Math.random();
      d.floor = floor;
    }
  }

  update(dt: number): void {
    for (let i = 0; i < this.count; i++) {
      const d = this.state[i];
      if (d.life <= 0) continue;
      d.life -= dt;
      d.v.z -= 4 * dt;
      d.p.addScaledVector(d.v, dt);
      // Landing: bounce, losing most of the speed, and stop turning
      if (d.p.z < d.floor + d.size / 2 && d.v.z < 0) {
        d.p.z = d.floor + d.size / 2;
        d.v.set(d.v.x * 0.4, d.v.y * 0.4, -d.v.z * 0.3);
        d.spin.multiplyScalar(0.4);
      }
      d.q.set(d.q.x + d.spin.x * dt, d.q.y + d.spin.y * dt, d.q.z + d.spin.z * dt);
      const scale = d.life <= 0 ? 0 : d.size * Math.min(1, d.life);
      this.matrix.compose(d.p, this.quaternion.setFromEuler(d.q), new THREE.Vector3(scale, scale, scale * 0.7));
      this.mesh.setMatrixAt(i, this.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}

/** Where the table stands in the station (its units): in front of the biggest hut. */
const TABLE_IN_STATION = [0.1, -0.2];

/**
 * A table outside the hut, 1 unit across, on z = 0: a bottle and glasses of
 * whisky, one of them with a piece of ice core in it, its bubbles rising.
 * Returns the group and the bubbles, moved in update().
 */
function whiskyTable(): { group: THREE.Group; bubbles: THREE.Mesh[] } {
  const group = new THREE.Group();
  const wood = new THREE.MeshMatcapMaterial({ matcap: matcap("#a0703f", "#3b2410") });
  const top = mesh(new THREE.BoxGeometry(1, 0.5, 0.04), wood);
  top.position.z = 0.42;
  group.add(top);
  for (const [x, y] of [[-0.45, -0.2], [0.45, -0.2], [-0.45, 0.2], [0.45, 0.2]]) {
    const leg = mesh(new THREE.BoxGeometry(0.04, 0.04, 0.4), wood);
    leg.position.set(x, y, 0.2);
    group.add(leg);
  }
  const glass = new THREE.MeshMatcapMaterial({ matcap: matcap("#eef7ff", "#8aa3b8"), transparent: true, opacity: 0.35, depthWrite: false });
  const whisky = new THREE.MeshMatcapMaterial({ matcap: matcap("#e9a43b", "#6e3608", "#ffe2a8") });
  const ice = new THREE.MeshMatcapMaterial({ matcap: matcap("#f2f9ff", "#93b7d6"), transparent: true, opacity: 0.85 });
  const tumbler = new THREE.CylinderGeometry(0.075, 0.065, 0.16, 20).rotateX(Math.PI / 2).translate(0, 0, 0.08);
  const drink = new THREE.CylinderGeometry(0.068, 0.063, 0.07, 20).rotateX(Math.PI / 2).translate(0, 0, 0.04);
  const glasses: [number, number][] = [[-0.25, 0.05], [0.05, -0.1], [0.3, 0.1]];
  const bubbles: THREE.Mesh[] = [];
  glasses.forEach(([x, y], k) => {
    const g = new THREE.Group();
    g.position.set(x, y, 0.44);
    g.add(mesh(tumbler, glass), mesh(drink, whisky));
    if (k === 1) {
      // Lorius's glass: a piece of core, the old air bubbling out of it
      const cube = mesh(new THREE.BoxGeometry(0.06, 0.06, 0.05), ice);
      cube.position.z = 0.07;
      cube.rotation.set(0.3, 0.2, 0.5);
      g.add(cube);
      const bubble = new THREE.SphereGeometry(0.008, 6, 4);
      const air = new THREE.MeshBasicMaterial({ color: "#ffffff", transparent: true, opacity: 0.8 });
      for (let i = 0; i < 12; i++) {
        const b = mesh(bubble, air);
        const a = i * 2.4;
        b.position.set(0.035 * Math.cos(a), 0.035 * Math.sin(a), 0.03 + ((i * 0.37) % 1) * 0.12);
        g.add(b);
        bubbles.push(b);
      }
    }
    group.add(g);
  });
  // The bottle, with a label
  const bottle = mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.26, 16).rotateX(Math.PI / 2).translate(0, 0, 0.13), whisky);
  const neck = mesh(new THREE.CylinderGeometry(0.02, 0.03, 0.1, 12).rotateX(Math.PI / 2).translate(0, 0, 0.31), whisky);
  const label = mesh(new THREE.CylinderGeometry(0.062, 0.062, 0.08, 16).rotateX(Math.PI / 2).translate(0, 0, 0.12), new THREE.MeshBasicMaterial({ color: "#f3ead2" }));
  for (const m of [bottle, neck, label]) {
    m.position.set(-0.05, 0.15, 0.44);
    group.add(m);
  }
  // Not drawn: a box around the table, so it can be hovered from afar
  const box = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.8, 1), new THREE.MeshBasicMaterial({ visible: false }));
  box.position.z = 0.4;
  group.add(box);
  return { group, bubbles };
}

/** The glass of Lorius, for the tooltip: whisky, a piece of core, and the old air rising from it. */
const WHISKY_GLASS = (() => {
  const bubbles = [
    [44, 0], [52, 0.6], [40, 1.1], [56, 1.7], [48, 2.3], [43, 2.9],
  ].map(([x, delay]) =>
    `<circle cx="${x}" cy="74" r="2.2" fill="#fff" stroke="#c98a2b" stroke-width="0.6">` +
    `<animate attributeName="cy" from="70" to="44" dur="3.4s" begin="${delay}s" repeatCount="indefinite"/>` +
    `<animate attributeName="opacity" values="0;1;1;0" dur="3.4s" begin="${delay}s" repeatCount="indefinite"/></circle>`,
  );
  return (
    `<svg width="96" height="96" viewBox="0 0 96 96" role="img" aria-label="A glass of whisky with a piece of ice core, bubbling">` +
    `<path d="M24 34 L72 34 L68 84 L28 84 Z" fill="#e9a43b" opacity="0.85"/>` +
    `<path d="M22 44 L74 44" stroke="#f6c66f" stroke-width="2"/>` +
    `<rect x="36" y="40" width="22" height="18" rx="3" fill="#eaf5ff" stroke="#93b7d6" stroke-width="1.5" transform="rotate(-14 47 49)" opacity="0.92"/>` +
    bubbles.join("") +
    `<path d="M18 14 L78 14 L72 88 L24 88 Z" fill="#dfeefa" fill-opacity="0.25" stroke="#7d8d9c" stroke-width="2.5" stroke-linejoin="round"/>` +
    `<path d="M24 22 L28 80" stroke="#fff" stroke-width="3" stroke-linecap="round" opacity="0.7"/>` +
    `</svg>`
  );
})();

/** A mesh at a place (x, y, z), scaled along its axes. */
function piece(geometry: THREE.BufferGeometry, material: THREE.Material, [x, y, z]: number[], [sx, sy, sz]: number[] = [1, 1, 1]): THREE.Mesh {
  const m = mesh(geometry, material);
  m.position.set(x, y, z);
  m.scale.set(sx, sy, sz);
  return m;
}

const SPHERE = new THREE.SphereGeometry(1, 24, 16);
const SNOW = new THREE.MeshMatcapMaterial({ matcap: matcap("#ffffff", "#a8b8c8") });

/** Random numbers from a seed: the same shapes each time. */
function seeded(seed: number): () => number {
  return () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
}

/**
 * Locks of hair over an ellipsoid (centre, radii), hanging: n thin cones out of its surface,
 * bent down, each about length long; none over the front (+x) where face is true.
 */
function fur(material: THREE.Material, centre: number[], radii: number[], n: number, length: number, seed: number, face = false): THREE.InstancedMesh {
  const lock = new THREE.ConeGeometry(0.35, 1, 5).translate(0, 0.5, 0);
  const locks = new THREE.InstancedMesh(lock, material, n);
  locks.frustumCulled = false;
  const random = seeded(seed);
  const [cx, cy, cz] = centre;
  const [rx, ry, rz] = radii;
  const up = new THREE.Vector3(0, 1, 0);
  const matrix = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  let k = 0;
  while (k < n) {
    const d = new THREE.Vector3(random() - 0.5, random() - 0.5, random() - 0.5);
    if (d.lengthSq() > 0.25 || d.lengthSq() < 0.01) continue;
    d.normalize();
    if (face && d.x > 0.45 && d.z > -0.5) continue;
    const at = new THREE.Vector3(cx + rx * d.x, cy + ry * d.y, cz + rz * d.z);
    const out = new THREE.Vector3(d.x / rx, d.y / ry, d.z / rz).normalize();
    const along = out.addScaledVector(new THREE.Vector3(0, 0, -1), 0.9 + 0.4 * random()).normalize();
    q.setFromUnitVectors(up, along);
    const l = length * (0.6 + 0.8 * random());
    matrix.compose(at.addScaledVector(along, -0.3 * l), q, new THREE.Vector3(l * 0.4, l, l * 0.4));
    locks.setMatrixAt(k++, matrix);
  }
  return locks;
}

/**
 * Fur, to wrap around the parts of the yeti: thousands of short strands, light and shadowed, hanging
 * down (along the meridians of a sphere), over white.
 */
function furTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 256;
  const context = canvas.getContext("2d")!;
  context.fillStyle = "#dfe5eb";
  context.fillRect(0, 0, 256, 256);
  const random = seeded(3);
  context.lineCap = "round";
  for (let k = 0; k < 6000; k++) {
    const [x, y] = [random() * 256, random() * 256];
    const length = 6 + 16 * random();
    const dark = random() < 0.35;
    context.strokeStyle = dark ? `rgba(105, 122, 140, ${0.25 + 0.3 * random()})` : `rgba(255, 255, 255, ${0.5 + 0.5 * random()})`;
    context.lineWidth = 0.8 + 1.4 * random();
    context.beginPath();
    context.moveTo(x, y);
    // Wavy, mostly down
    context.quadraticCurveTo(x + 4 * (random() - 0.5), y + length / 2, x + 6 * (random() - 0.5), y + length);
    context.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(3, 2);
  return texture;
}

/** One arm of the yeti, hanging from the shoulder: the elbow and the hand turn about y. */
interface Arm {
  shoulder: THREE.Group;
  elbow: THREE.Group;
  hand: THREE.Mesh;
}

/**
 * The yeti of Yetisports, 1 unit high, facing +x, on z = 0: tall and slim
 * like a man, very hairy, two dark eyes and a small mouth. The body turns at
 * the hips (body: lean rotation.y, twist rotation.z), the arms at the
 * shoulders and the elbows (rotation.y); a snowball in the right hand (at -y).
 */
function yeti(): { group: THREE.Group; body: THREE.Group; right: Arm; left: Arm; ball: THREE.Mesh } {
  const group = new THREE.Group();
  // Fur all over: a texture of strands on the body, and thick locks hanging out of it
  const strands = furTexture();
  const skin = new THREE.MeshMatcapMaterial({ matcap: matcap("#f4f7fa", "#7f8d9b"), map: strands });
  const hair = new THREE.MeshMatcapMaterial({ matcap: matcap("#fbfcfd", "#8d9aa8"), map: strands });
  const black = new THREE.MeshBasicMaterial({ color: "#121518" });
  const limb = (length: number, radius: number) => new THREE.CapsuleGeometry(radius, length, 4, 12).rotateX(Math.PI / 2).translate(0, 0, -length / 2);
  let seed = 11;
  // A hairy part: an ellipsoid with its locks
  const part = (parent: THREE.Object3D, centre: number[], radii: number[], locks: number, face = false) => {
    parent.add(piece(SPHERE, skin, centre, radii));
    parent.add(fur(hair, centre, radii, locks * 3, 0.085, seed++, face));
  };
  // Legs and big feet
  for (const y of [-0.075, 0.075]) {
    part(group, [0, y, 0.27], [0.06, 0.06, 0.22], 50);
    group.add(piece(SPHERE, skin, [0.05, y * 1.1, 0.025], [0.11, 0.06, 0.03]));
  }
  // From the hips up: the chest, the head with its eyes and mouth
  const body = new THREE.Group();
  body.position.z = 0.47;
  group.add(body);
  part(body, [0, 0, 0.04], [0.09, 0.11, 0.08], 50);
  part(body, [0, 0, 0.2], [0.1, 0.15, 0.15], 140);
  part(body, [0.01, 0, 0.42], [0.08, 0.075, 0.09], 70, true);
  for (const y of [-0.028, 0.028]) body.add(piece(SPHERE, black, [0.082, y, 0.44], [0.012, 0.012, 0.016]));
  body.add(piece(SPHERE, black, [0.083, 0, 0.385], [0.006, 0.02, 0.007]));
  // Arms: upper arm, forearm, a big hand
  const arm = (side: number): Arm => {
    const shoulder = new THREE.Group();
    shoulder.position.set(0, 0.17 * side, 0.3);
    shoulder.add(new THREE.Mesh(limb(0.17, 0.045), skin), fur(hair, [0, 0, -0.1], [0.05, 0.05, 0.12], 120, 0.07, seed++));
    const elbow = new THREE.Group();
    elbow.position.z = -0.2;
    elbow.add(new THREE.Mesh(limb(0.16, 0.04), skin), fur(hair, [0, 0, -0.09], [0.045, 0.045, 0.1], 100, 0.07, seed++));
    const hand = piece(SPHERE, skin, [0.01, 0, -0.22], [0.045, 0.04, 0.05]);
    elbow.add(hand);
    shoulder.add(elbow);
    body.add(shoulder);
    shoulder.traverse((o) => (o.frustumCulled = false));
    return { shoulder, elbow, hand };
  };
  const right = arm(-1);
  const left = arm(1);
  const ball = piece(SPHERE, SNOW, [0.03, 0, -0.27], [0.06, 0.06, 0.06]);
  right.elbow.add(ball);
  // More snowballs, ready
  for (const [x, y, z] of [[0.22, -0.28, 0.05], [0.3, -0.2, 0.05], [0.28, -0.36, 0.05], [0.27, -0.28, 0.13]]) {
    group.add(piece(SPHERE, SNOW, [x, y, z], [0.055, 0.055, 0.055]));
  }
  return { group, body, right, left, ball };
}

/**
 * A penguin, 1 unit high, facing +x, on z = 0. Its flippers turn at the shoulders (userData.flippers,
 * rotation.x), its feet at the ankles (userData.feet, rotation.y).
 */
function penguin(): THREE.Group {
  const group = new THREE.Group();
  const black = new THREE.MeshMatcapMaterial({ matcap: matcap("#3a4048", "#06080a", "#7c8794") });
  const white = new THREE.MeshMatcapMaterial({ matcap: matcap("#ffffff", "#a9b4bf") });
  const orange = new THREE.MeshMatcapMaterial({ matcap: matcap("#ffb340", "#a04a00", "#ffe0a0") });
  const eye = new THREE.MeshBasicMaterial({ color: "#ffffff" });
  const pupil = new THREE.MeshBasicMaterial({ color: "#000000" });
  group.add(piece(SPHERE, black, [0, 0, 0.42], [0.24, 0.24, 0.38]));
  group.add(piece(SPHERE, white, [0.07, 0, 0.38], [0.19, 0.19, 0.32]));
  group.add(piece(SPHERE, black, [0.01, 0, 0.82], [0.15, 0.15, 0.14]));
  for (const y of [-0.06, 0.06]) {
    group.add(piece(SPHERE, eye, [0.12, y, 0.86], [0.035, 0.035, 0.035]));
    group.add(piece(SPHERE, pupil, [0.152, y, 0.86], [0.012, 0.016, 0.016]));
    // The foot from the ankle, to kick (userData.feet, rotation.y)
    const ankle = new THREE.Group();
    ankle.position.set(0, y * 1.5, 0.04);
    ankle.add(piece(SPHERE, orange, [0.08, 0, -0.02], [0.09, 0.05, 0.025]));
    group.add(ankle);
    (group.userData.feet ??= []).push(ankle);
  }
  group.add(piece(new THREE.ConeGeometry(0.035, 0.12, 10).rotateZ(-Math.PI / 2), orange, [0.18, 0, 0.8]));
  const flippers = [-1, 1].map((side) => {
    const shoulder = new THREE.Group();
    shoulder.position.set(0, 0.23 * side, 0.62);
    shoulder.add(piece(SPHERE, black, [0.01, 0.025 * side, -0.2], [0.08, 0.03, 0.24]));
    shoulder.userData.side = side;
    group.add(shoulder);
    return shoulder;
  });
  group.userData.flippers = flippers;
  // Turned about its own axes first (lying down to swim, rocking), then towards where it goes
  group.rotation.order = "ZYX";
  return group;
}

/** Spread the flippers of a penguin (rad out from its sides). */
function flap(penguin: THREE.Group, angle: number): void {
  for (const f of penguin.userData.flippers as THREE.Group[]) f.rotation.x = -f.userData.side * angle;
}

/** An orca, 1 unit long along +x (head at +x), its middle at z = 0. Its tail flukes turn up and down (tail, rotation.y). */
function orca(): { group: THREE.Group; tail: THREE.Group } {
  const group = new THREE.Group();
  const black = new THREE.MeshMatcapMaterial({ matcap: matcap("#3a4048", "#06080a", "#8a96a3") });
  const white = new THREE.MeshMatcapMaterial({ matcap: matcap("#ffffff", "#aab5c0") });
  group.add(piece(SPHERE, black, [0.05, 0, 0], [0.42, 0.11, 0.11]));
  group.add(piece(SPHERE, black, [-0.3, 0, 0.01], [0.18, 0.055, 0.06]));
  group.add(piece(SPHERE, white, [0.12, 0, -0.035], [0.3, 0.085, 0.08]));
  for (const y of [-0.07, 0.07]) {
    group.add(piece(SPHERE, white, [0.3, y, 0.03], [0.06, 0.03, 0.025]));
    const fin = piece(SPHERE, black, [0.18, y * 1.7, -0.06], [0.07, 0.05, 0.012]);
    fin.rotation.x = -Math.sign(y) * 0.4;
    group.add(fin);
  }
  // The tall dorsal fin
  const shape = new THREE.Shape([new THREE.Vector2(0.06, 0), new THREE.Vector2(-0.06, 0), new THREE.Vector2(-0.04, 0.2)]);
  const dorsal = new THREE.ExtrudeGeometry(shape, { depth: 0.014, bevelEnabled: false }).translate(0, 0, -0.007).rotateX(Math.PI / 2);
  group.add(piece(dorsal, black, [0, 0, 0.09]));
  const tail = new THREE.Group();
  tail.position.set(-0.46, 0, 0.01);
  for (const y of [-0.06, 0.06]) {
    const fluke = piece(SPHERE, black, [-0.05, y, 0], [0.06, 0.07, 0.012]);
    fluke.rotation.z = -Math.sign(y) * 0.6;
    tail.add(fluke);
  }
  group.add(tail);
  return { group, tail };
}

/**
 * An iceberg, about 1 unit long along x (5 km for ICEBERG_SIZE 5), its waterline at z = 0: the hull
 * of jagged points, sharp flat faces. A flat face towards -x (x = -0.5) with the target painted on
 * it, centred at z = TARGET_HEIGHT.
 */
function iceberg(): THREE.Group {
  const group = new THREE.Group();
  const random = seeded(5);
  const points: THREE.Vector3[] = [];
  // The face of the target
  for (const [y, z] of [[-0.33, -0.06], [0.33, -0.08], [-0.3, 0.6], [0.28, 0.62]]) points.push(new THREE.Vector3(-0.5, y, z));
  // Rings from under the water to the top, narrowing, uneven; then a few peaks
  for (const [z, rx, ry, n] of [[-0.3, 0.45, 0.7, 9], [0, 0.55, 0.82, 11], [0.35, 0.5, 0.72, 10], [0.62, 0.36, 0.55, 8]]) {
    for (let k = 0; k < n; k++) {
      const a = ((k + 0.6 * random()) / n) * Math.PI * 2;
      const f = 0.72 + 0.3 * random();
      points.push(new THREE.Vector3(Math.max(-0.48, rx * f * Math.cos(a)), ry * f * Math.sin(a), z + 0.12 * (random() - 0.5)));
    }
  }
  for (const [x, y, z] of [[0.12, 0.25, 1.0], [0.25, -0.2, 0.82], [-0.15, 0.45, 0.78], [0.3, 0.5, 0.7]]) points.push(new THREE.Vector3(x, y, z));
  const ice = new THREE.MeshMatcapMaterial({ matcap: matcap("#f2faff", "#6f98bb"), flatShading: true });
  group.add(mesh(new ConvexGeometry(points), ice));
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const context = canvas.getContext("2d")!;
  for (let k = 5; k > 0; k--) {
    context.fillStyle = k % 2 ? "#d42a2a" : "#ffffff";
    context.beginPath();
    context.arc(64, 64, (64 * k) / 5, 0, Math.PI * 2);
    context.fill();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const target = new THREE.CircleGeometry(0.24, 48).rotateY(-Math.PI / 2);
  group.add(piece(target, new THREE.MeshBasicMaterial({ map: texture }), [-0.502, 0, TARGET_HEIGHT]));
  return group;
}

export interface Egg {
  /** Stable: what the browser remembers as found (eggcounter.ts) */
  id: string;
  title: string;
  rows: [string, string][];
  note: string;
  objects: THREE.Object3D[];
  /** A drawing for the tooltip (SVG markup), if any */
  picture?: string;
  /** Km from the camera: further away, it can't be hovered or collected. */
  reach: number;
  /** The view to fly to it: km from the camera, and angle above the horizon (degrees), from the side of the home view */
  view: [number, number];
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
const CROMLECH_SIZE = 14; // a few metres across in reality: made much bigger, to be seen at all
const CROMLECH_AT = polarStereographic(-75.42, 122.45); // about 13 km south of the Little Dome C drill site, clear of its marker
const STATION_SIZE = 10; // about 500 m across in reality: made bigger, like the cromlech
const STATION_AT = polarStereographic(-66.6, 139.99); // Dumont d'Urville, Île des Pétrels, moved 7 km out to sea, clear of the ice
const TABLE_SIZE = 0.9; // a bit out of scale with the huts, to see the glasses
const TRAIN_AT = polarStereographic(-79.741, -78.5448); // the Skytrain Ice Rise drill site, inside the loop
// Distance (km) of the track from the drill site, every 5° from +x anticlockwise: about 8 km out
// on the Ronne Ice Shelf from the edge of the grounded ice rise in Bedmap3, smoothed; across the
// grounded ice where the rise joins the land (about 235° to 325°), where it goes straight through
// the ice instead, from the height of the shelf on one side to the other
const TRACK_RADII = [
  40.9, 41.7, 42.5, 43.3, 44.2, 45.2, 46.4, 47.8, 49.1, 50.3, 51.2, 51.9, 52.5, 53.3, 54.1, 55.2, 56.5, 57.8, 59.1, 60.5, 61.9, 63.5,
  64.7, 65.8, 66.8, 67.5, 67.6, 67.5, 66.9, 65.9, 64.9, 63.9, 63.3, 63.0, 63.1, 63.5, 64.5, 65.8, 67.6, 69.3, 70.1, 69.4, 67.1, 63.2,
  59.0, 55.6, 53.0, 51.5, 50.9, 50.4, 49.6, 48.8, 48.0, 47.2, 46.4, 45.6, 44.8, 44.0, 43.2, 42.4, 41.6, 40.7, 39.6, 38.6, 37.8, 37.2,
  37.1, 37.3, 37.8, 38.4, 39.2, 40.0,
];
const TRAIN_CARS = 14;
const TRAIN_HEIGHT = 0.9; // km, about a car with its roof
const CAR_LENGTH = 2; // km, with the gap between cars
const TRAIN_SPEED = 0.15; // rad/s around the loop: a lap in about 40 s (a year in the film)
const TRACK_SAMPLES = 720;
const TUNNEL = [232.5, 327.5].map((d) => Math.round((d / 360) * TRACK_SAMPLES)); // samples of the grounded part

// Orca slap: the island of the yeti in Pine Island Bay, off the Pine Island and Thwaites glaciers (not
// in Bedmap3, open sea there), the island where the penguins wait, and the iceberg with the target,
// out in the bay (towards ICEBERG_AT). In the frame of the scene (units of SLAP_SCALE km): from the
// middle of the island of the yeti, +x towards the iceberg
const SLAP_AT = polarStereographic(-74.345, -103.346);
const ICEBERG_AT = polarStereographic(-74.484, -104.061);
const SLAP_SCALE = 0.6; // km per unit of the scene
const ICEBERG_DISTANCE = 26;
const SLAP_ISLAND = 7; // across the top
const WAIT_ISLAND = 4;
const WAIT_AT = new THREE.Vector3(2, -13, 0);
const YETI_SIZE = 2.2; // all out of scale, like in the game
const YETI_IN_SCENE = [0.8, 1.4];
const PENGUIN_SIZE = 0.9;
const PENGUINS = 6;
// The way of a penguin: from the island where they wait, into the sea, swimming to the island of the
// yeti, out onto it, to its edge by the orca; the others coming back from the far side
const TO_YETI = WAIT_AT.clone().negate().normalize();
const DIVE_FROM = WAIT_AT.clone().addScaledVector(TO_YETI, 1.3);
const DIVE_TO = WAIT_AT.clone().addScaledVector(TO_YETI, 3.2);
const SWIM_TO = TO_YETI.clone().multiplyScalar(-5);
const HOP_TO = TO_YETI.clone().multiplyScalar(-2.7);
const POP_FROM = WAIT_AT.clone().addScaledVector(TO_YETI, -3.3);
// Where the others wait, around the middle of their island (one for each but the one playing)
const SPOTS = [[0.3, 0.5], [-0.6, 0.2], [0.1, -0.6], [-0.5, -0.6], [0.7, -0.2]].map(([x, y]) => WAIT_AT.clone().add(new THREE.Vector3(x, y, 0)));
const LYING = [1, 3]; // spots where they lie on the ground
const EDGE = new THREE.Vector3(3, -0.5, 0); // where the penguin jumps from
const ORCA_SIZE = 4.5;
const ORCA_X = 7.7; // facing the yeti, its head just off the island
const APEX = new THREE.Vector3(ORCA_X + 2.4, 0, 6.5); // where the snowball hits the penguin
const ICEBERG_SIZE = 5;
const TARGET_HEIGHT = 0.28; // of the iceberg, above the waterline
const SLAP_LOOP = 12.4; // s, one penguin

/** Distance (km) of the track from the drill site in a direction (rad). */
function trackRadius(a: number): number {
  const f = ((((a / (Math.PI * 2)) % 1) + 1) % 1) * TRACK_RADII.length;
  const i = Math.floor(f);
  return TRACK_RADII[i] + (f - i) * (TRACK_RADII[(i + 1) % TRACK_RADII.length] - TRACK_RADII[i]);
}

export class EasterEggs {
  readonly group = new THREE.Group();
  private exaggeration = 1;
  private vessel = vessel();
  private fossil = fossil();
  private cromlech = cromlech();
  // Cores of the cromlech knocked over by the mouse: falling, then broken on the snow (until a reload)
  private falling: { core: THREE.Mesh; toward: number; angle: number; speed: number }[] = [];
  private sliding: { piece: THREE.Mesh; from: THREE.Vector3; to: THREE.Vector3; t: number }[] = [];
  // Where each thing could be put (terrain under it), apart from being drawn or not (cull())
  private placed = new Map<THREE.Object3D, boolean>();
  private station = station();
  private flag = this.station.getObjectByName("flag") as THREE.Mesh;
  private time = 0;
  private table = whiskyTable();
  private island = island();
  // The train: its cars, the track, and where they are on the loop
  private train = new THREE.Group();
  private cars = train(TRAIN_CARS);
  private track = new THREE.LineLoop(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: "#7d8794", transparent: true, opacity: 0.4, depthWrite: false }),
  );
  // Points of the track (km, from the drill site; z with the exaggeration), every 360/TRACK_SAMPLES degrees
  private trackPoints = new Float32Array(TRACK_SAMPLES * 3);
  private trainAngle = 0;
  // Snow thrown up where the train goes into the ice and out of it
  private snow = new SnowBursts();
  private debris = new IceDebris();
  // Track samples where the train goes deep into the ice and comes out (from the surface, in place())
  private portals: number[] = [];
  private portalTops: number[] = [];
  // Orca slap: the scene turned towards the iceberg, its island, the players, the snowball in the air
  private slap = new THREE.Group();
  private slapIsland = island(true);
  private waitIsland = island(true);
  private slapTop = 0;
  private yeti = yeti();
  private orca = orca();
  private penguins = Array.from({ length: PENGUINS }, penguin);
  private iceberg = iceberg();
  private snowball = piece(SPHERE, SNOW, [0, 0, 0], [0.25, 0.25, 0.25]);
  private puffs = new SnowBursts(600, 0.6 * SLAP_SCALE);
  private releaseAt = new THREE.Vector3(); // where the snowball left the hand of the yeti
  private slapBefore = 0;
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
    this.cromlech.scale.setScalar(CROMLECH_SIZE);
    this.station.scale.setScalar(STATION_SIZE);
    this.table.group.scale.setScalar(TABLE_SIZE);
    for (const car of this.cars) car.scale.setScalar(CAR_LENGTH);
    this.train.add(this.track, ...this.cars, this.snow.points, this.debris.mesh);
    this.track.frustumCulled = false;
    // Faint, and gone from afar: not to give the train away
    this.track.onBeforeRender = (_renderer, _scene, camera) => {
      const d = camera.position.distanceTo(this.train.position);
      (this.track.material as THREE.LineBasicMaterial).opacity = 0.4 * THREE.MathUtils.clamp((400 - d) / 200, 0, 1);
    };
    this.yeti.group.scale.setScalar(YETI_SIZE);
    this.yeti.group.position.set(YETI_IN_SCENE[0], YETI_IN_SCENE[1], 0);
    this.yeti.group.rotation.z = Math.atan2(APEX.y - YETI_IN_SCENE[1], APEX.x - YETI_IN_SCENE[0]);
    this.orca.group.scale.setScalar(ORCA_SIZE);
    this.orca.group.position.x = ORCA_X;
    this.orca.group.rotation.z = Math.PI;
    this.waitIsland.position.copy(WAIT_AT);
    this.slap.scale.setScalar(SLAP_SCALE);
    for (const p of this.penguins) p.scale.setScalar(PENGUIN_SIZE);
    this.iceberg.scale.setScalar(ICEBERG_SIZE);
    this.slap.add(this.slapIsland, this.waitIsland, this.yeti.group, this.orca.group, ...this.penguins, this.iceberg, this.snowball, this.puffs.points);
    this.group.add(this.vessel, this.fossil, this.whirlpool, this.cromlech, this.island, this.station, this.table.group, this.train, this.slap);
    this.fossilRows = [
      ["Position", "69.7° S, 152.8° E, at the bottom of a deep trough"],
      ["Near", "the lines of Bodart and Sutter (2025b)"],
      ["Age", "older than the oldest isochrone"],
      ["Searched by", "Johannes Sutter, still on the quest to find it"],
    ];
    this.eggs = [
      {
        id: "vessel",
        title: "Unidentified crashed vessel",
        rows: [
          ["Position", "84.4° S, 165.6° E, Mount Kirkpatrick"],
          ["Origin", "unknown"],
          ["Crew", "not found"],
        ],
        note: "Not in Bedmap3. Lights still on.",
        objects: [this.vessel],
        view: [60, 35],
        reach: 300,
      },
      {
        id: "fossil",
        title: "Fossil of Charybdis",
        rows: this.fossilRows,
        note: "The sea monster of Greek myth (Homer, Odyssey XII), frozen in the basal ice. You found it before Johannes did.",
        objects: [this.fossil],
        view: [45, 80],
        reach: 300,
      },
      {
        id: "whirlpool",
        title: "Remnant whirlpool of Charybdis",
        rows: [
          ["Position", "64° S, 150° E, off the Wilkes Subglacial Basin"],
          ["Depth", "all the way down to the seafloor, about 3.6 km"],
          ["Activity", "swallows the sea three times a day, and spits it out again"],
          ["Last seen", "by Odysseus, who survived by clinging to a fig tree"],
        ],
        note: "Left over from the activity of Charybdis, before she settled inland under the ice. Still spinning.",
        objects: [this.whirlpool],
        view: [300, 55],
        reach: 1500,
      },
      {
        id: "cromlech",
        title: "Ice core cromlech",
        rows: [
          ["Position", "75.3° S, 122.4° E, next to Little Dome C"],
          ["Stones", "about 50 ice cores, 1 m long, stood on end in a circle"],
          ["Raised", "during the 2023–2024 drilling season"],
          ["Builders", "a team of women scientists of the Beyond EPICA camp"],
          ["Rites", "dancing and screaming out loud in the middle, to decompress from the long working hours at the camp"],
          ["State", "some cores broken, the pieces lying around"],
        ],
        note: "The first stone circle of Antarctica, made of ice, where its builders danced and screamed in the middle. Not in Bedmap3.",
        objects: [this.cromlech],
        view: [60, 40],
        reach: 150,
      },
      {
        id: "lorius",
        title: "Claude Lorius's glass of whisky",
        rows: [
          ["Position", "66.7° S, 140.0° E, Dumont d'Urville station, Adélie Land"],
          ["When", "winter 1965, when he was in charge of the station"],
          ["On the rocks", "a piece of the first ice cores drilled in the coastal ice nearby"],
          ["Seen", "the air bubbles bursting out of the ice as it melted"],
          ["Idea", "the bubbles hold the air of the time when the snow fell: an archive of past atmospheres"],
          ["20 years later", "the Vostok core: CO2 and Antarctic temperature going together over 150,000 years (Nature, 1987)"],
        ],
        note: "Claude Lorius (1932–2023), French glaciologist. After a hard day of drilling, he dropped a piece of core into his well-earned whisky and watched the old air escape. Still bubbling.",
        objects: [this.table.group],
        view: [7, 25],
        picture: WHISKY_GLASS,
        reach: 250,
      },
      {
        id: "snowpiercer",
        title: "Snowpiercer",
        rows: [
          ["Position", "79.7° S, 78.5° W, around Skytrain Ice Rise"],
          ["Track", "a loop of about 340 km around the ice rise, on the flat Ronne Ice Shelf but where the rise joins the land"],
          ["Lap", "once a year in the film; here a bit faster"],
          ["Engine", "the Sacred Eternal Engine, never stopped"],
          ["Passengers", "the last survivors of a geoengineering that froze the Earth, first class at the front, the tail at the back"],
          ["From", "Le Transperceneige, a French comic by Jacques Lob and Jean-Marc Rochette (1982), and the film of Bong Joon-ho (2013)"],
        ],
        note: "Someone read \"Skytrain\" a bit too literally. Do not pull the emergency brake.",
        objects: [this.train],
        view: [320, 50],
        reach: 400,
      },
      {
        id: "orcaslap",
        title: "Orca Slap",
        rows: [
          ["Position", "74.3° S, 103.3° W, Pine Island Bay, off the Pine Island and Thwaites glaciers"],
          ["Game", "Yetisports Part 2: Orca Slap, by Chris Hilgert (2004)"],
          ["Rules", "a penguin jumps on the orca, the orca slaps it up with its tail, the yeti hits it with a snowball, towards the target on the iceberg"],
          ["Players", "one yeti, one orca, and a queue of penguins that never ends"],
          ["Target", "on an iceberg about 16 km out in the bay"],
        ],
        note: "No penguin was hurt: they swim back and queue again. Not in Bedmap3.",
        objects: [this.slap],
        view: [35, 22],
        reach: 300,
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
    this.placed.set(this.vessel, ground !== null);
    // Half buried
    if (ground !== null) this.vessel.position.set(vx, vy, z(ground) - 0.1 * VESSEL_SIZE);
    // On the bed (moved onto it in the shader), partly sunk into it
    const [fx, fy] = FOSSIL_AT;
    const floor = this.bedAt(fx, fy);
    this.placed.set(this.fossil, floor !== null);
    this.fossil.position.set(fx, fy, -FOSSIL_BURIED * FOSSIL_SIZE * FOSSIL_RELIEF);
    const surface = this.surfaceAt(fx, fy);
    const ice = surface === null || floor === null ? null : Math.round((surface - floor) / 10) * 10;
    this.fossilRows.splice(4, 1, ...(ice === null ? [] : [["Ice above", `about ${ice} m`] as [string, string]]));

    // Standing on the ice surface, the bottom of the cores a little sunk into it
    const [cx, cy] = CROMLECH_AT;
    const snow = this.surfaceAt(cx, cy);
    this.placed.set(this.cromlech, snow !== null);
    if (snow !== null) this.cromlech.position.set(cx, cy, z(snow) - 0.01 * CROMLECH_SIZE);

    // On the island, its top a bit above the sea or the ice around, its foot on the seafloor
    const [sx, sy] = STATION_AT;
    const top = z(Math.max(0, this.heightAt(sx, sy) ?? 0) + 30);
    const foot = z(Math.min(0, this.bedAt(sx, sy) ?? -300) - 50);
    this.island.position.set(sx, sy, top);
    this.island.scale.set(STATION_SIZE, STATION_SIZE, top - foot);
    this.station.position.set(sx, sy, top);
    const [tx, ty] = TABLE_IN_STATION;
    this.table.group.position.set(sx + tx * STATION_SIZE, sy + ty * STATION_SIZE, top);

    // The track on the ice around Skytrain, a little above it; the cars follow it in update()
    const [rx, ry] = TRAIN_AT;
    const centre = this.surfaceAt(rx, ry);
    this.placed.set(this.train, centre !== null);
    this.train.position.set(rx, ry, z(centre ?? 0));
    const surfaceZ = new Float32Array(TRACK_SAMPLES);
    for (let k = 0; k < TRACK_SAMPLES; k++) {
      const a = (k / TRACK_SAMPLES) * Math.PI * 2;
      const r = trackRadius(a);
      const [x, y] = [r * Math.cos(a), r * Math.sin(a)];
      // Relative to the centre, as the cars
      const h = z(this.surfaceAt(rx + x, ry + y) ?? centre ?? 0) - z(centre ?? 0) + 0.02;
      this.trackPoints.set([x, y, h], k * 3);
      surfaceZ[k] = h;
    }
    // Through the grounded ice: in a straight slope between the shelf on both sides
    const [t0, t1] = TUNNEL;
    const [h0, h1] = [this.trackPoints[3 * t0 + 2], this.trackPoints[3 * t1 + 2]];
    for (let k = t0 + 1; k < t1; k++) this.trackPoints[3 * k + 2] = h0 + ((k - t0) / (t1 - t0)) * (h1 - h0);
    // Where the whole train is under the ice: deeper than a car is high
    const deep: number[] = [];
    for (let k = t0; k <= t1; k++) if (surfaceZ[k] - this.trackPoints[3 * k + 2] > TRAIN_HEIGHT) deep.push(k);
    this.portals = deep.length ? [deep[0], deep[deep.length - 1]] : [];
    this.portalTops = this.portals.map((k) => surfaceZ[k]);
    this.track.geometry.setAttribute("position", new THREE.BufferAttribute(this.trackPoints.slice(), 3));
    this.moveTrain();

    // The island of the yeti, its top a bit above the sea, its foot on the seafloor; the scene
    // turned so the penguins fly along +x, to the iceberg
    const [ix, iy] = SLAP_AT;
    const [bx, by] = ICEBERG_AT;
    const seafloor = this.bedAt(ix, iy);
    this.placed.set(this.slap, seafloor !== null);
    this.slap.position.set(ix, iy, 0);
    this.slap.rotation.z = Math.atan2(by - iy, bx - ix);
    // Heights in units of the scene, not shrunk with it: down to the seafloor
    this.slapTop = z(30) / SLAP_SCALE;
    const tall = this.slapTop - z(Math.min(0, seafloor ?? -300) - 50) / SLAP_SCALE;
    this.slapIsland.position.z = this.waitIsland.position.z = this.slapTop;
    this.slapIsland.scale.set(SLAP_ISLAND, SLAP_ISLAND, tall);
    this.waitIsland.scale.set(WAIT_ISLAND, WAIT_ISLAND, tall);
    this.yeti.group.position.z = this.slapTop;
    this.iceberg.position.x = ICEBERG_DISTANCE;
    this.playSlap();

    // From the sea surface down to the seafloor
    const [wx, wy] = WHIRLPOOL_AT;
    const depth = this.bedAt(wx, wy);
    this.placed.set(this.whirlpool, depth !== null && depth < 0);
    if (depth === null || depth >= 0) return;
    this.whirlpool.position.set(wx, wy, z(depth));
    this.whirlpool.scale.set(WHIRLPOOL_RADIUS, WHIRLPOOL_RADIUS, -z(depth));
  }

  /** Turn the whirlpool, raise the bubbles and move the train, by dt seconds. */
  update(dt: number): void {
    this.time += dt;
    if (this.station.visible) waveFlag(this.flag, this.time);
    const before = this.trainAngle;
    this.trainAngle = (this.trainAngle + TRAIN_SPEED * dt) % (Math.PI * 2);
    this.moveTrain();
    this.burstSnow(before);
    this.snow.update(dt);
    this.debris.update(dt);
    this.moveCores(dt);
    if (this.slap.visible) this.playSlap();
    this.puffs.update(dt);
    // The old air rising in the glass of Lorius, again and again
    for (const b of this.table.bubbles) b.position.z = 0.03 + ((b.position.z - 0.03 + 0.04 * dt) % 0.12);
    const map = this.whirlpool.material.map!;
    map.offset.x = (map.offset.x + 0.08 * dt) % 1;
    map.offset.y = (map.offset.y - 0.04 * dt) % 1;
  }

  /** The cars along the loop, one behind the other, on the track and facing along it. */
  private moveTrain(): void {
    const p = this.trackPoints;
    // A point of the track at an angle (rad), between the samples
    const at = (a: number, out: THREE.Vector3) => {
      const f = ((((a / (Math.PI * 2)) % 1) + 1) % 1) * TRACK_SAMPLES;
      const i = Math.floor(f) % TRACK_SAMPLES;
      const j = (i + 1) % TRACK_SAMPLES;
      const t = f - Math.floor(f);
      return out.set(p[3 * i] + t * (p[3 * j] - p[3 * i]), p[3 * i + 1] + t * (p[3 * j + 1] - p[3 * i + 1]), p[3 * i + 2] + t * (p[3 * j + 2] - p[3 * i + 2]));
    };
    const ahead = new THREE.Vector3();
    let a = this.trainAngle;
    for (const car of this.cars) {
      at(a, car.position);
      // Anticlockwise: heading along the track
      at(a + 0.002, ahead);
      car.rotation.z = Math.atan2(ahead.y - car.position.y, ahead.x - car.position.x);
      a -= CAR_LENGTH / trackRadius(a);
    }
  }

  /**
   * A burst of snow and blocks of ice where the engine goes deep into the ice, and where it comes
   * out again: once each time it passes (from the angle before to the angle now).
   */
  private burstSnow(before: number): void {
    const p = this.trackPoints;
    for (const sample of this.portals) {
      const portal = (sample / TRACK_SAMPLES) * Math.PI * 2;
      const passed = (a: number) => Math.atan2(Math.sin(a - portal), Math.cos(a - portal)) >= 0;
      if (passed(before) || !passed(this.trainAngle)) continue;
      const at = new THREE.Vector3(p[3 * sample], p[3 * sample + 1], p[3 * sample + 2]);
      const j = (sample + 1) % TRACK_SAMPLES;
      const along = new THREE.Vector3(p[3 * j] - at.x, p[3 * j + 1] - at.y, 0).normalize();
      // From the ice surface above the engine, where it breaks through
      const top = this.portalTops[this.portals.indexOf(sample)];
      at.z = top;
      this.snow.emit(at, along, 1500);
      this.debris.emit(at, along, 80, top);
    }
  }

  /**
   * One round of Orca Slap, from the time in the loop (SLAP_LOOP s, one penguin each): a penguin
   * leaves the others on their island, swims to the island of the yeti and jumps on the tail of the
   * orca, which slaps it up into the air; the yeti throws a snowball at it, which sends it flying to
   * the iceberg, flat on the target. It slides down into the sea, and pops out onto the island of the
   * others in the next round, from the far side, as the next one goes.
   */
  private playSlap(): void {
    const t = this.time % SLAP_LOOP;
    const round = Math.floor(this.time / SLAP_LOOP);
    const s = (from: number, to: number) => THREE.MathUtils.clamp((t - from) / (to - from), 0, 1);
    const ease = (x: number) => x * x * (3 - 2 * x);
    const passed = (at: number) => (this.slapBefore < at && t >= at) || (this.slapBefore > t && t >= at);
    const top = this.slapTop;
    this.slap.updateMatrixWorld(true);

    // The orca: breathing; its tail pressed down by the penguin, then whipped up
    this.orca.group.position.z = 0.1 + 0.05 * Math.sin(this.time * 1.7);
    const tail =
      t < 5.65 ? -0.35 * ease(s(5.4, 5.65)) : t < 5.85 ? -0.35 + 1.65 * s(5.65, 5.85) : 1.3 * (1 - ease(s(5.85, 6.6)));
    this.orca.tail.rotation.y = tail;
    this.orca.tail.updateMatrixWorld(true);
    const onTail = this.slap.worldToLocal(this.orca.tail.localToWorld(new THREE.Vector3(-0.06, 0, 0.012)));

    // The yeti: taking momentum, leaning back and turning, the arm cocked behind its head, the other
    // one pointing at the penguin; then the whole body thrown forward with the arm, letting go of the
    // snowball over its head; back up, and a new snowball from the pile
    const keys = (frames: number[][]) => {
      for (let i = 1; i < frames.length; i++) {
        const [[t0, v0], [t1, v1]] = [frames[i - 1], frames[i]];
        if (t < t1) return t < t0 ? v0 : v0 + (v1 - v0) * ease((t - t0) / (t1 - t0));
      }
      return frames[frames.length - 1][1];
    };
    const { body, right, left } = this.yeti;
    const idle = t < 5.4 || t > 7.8 ? 1 : 0;
    body.rotation.y = keys([[5.4, 0], [6.1, -0.35], [6.35, -0.4], [6.6, 0.35], [6.85, 0.5], [7.7, 0]]) + 0.03 * idle * Math.sin(this.time * 1.3);
    body.rotation.z = keys([[5.4, 0], [6.2, -0.6], [6.35, -0.65], [6.6, 0.45], [6.85, 0.55], [7.8, 0]]) + 0.08 * idle * Math.sin(this.time * 0.7);
    body.position.z = keys([[5.4, 0.47], [6.2, 0.43], [6.55, 0.45], [6.85, 0.42], [7.7, 0.47]]) + 0.004 * Math.sin(this.time * 2.2);
    right.shoulder.rotation.y = keys([[5.4, 0.15], [6.25, 2.6], [6.35, 2.75], [6.85, 5.4], [7.8, Math.PI * 2 + 0.15]]) + 0.06 * idle * Math.sin(this.time * 1.1);
    right.elbow.rotation.y = keys([[5.4, -0.3], [6.25, -1.6], [6.35, -1.7], [6.65, 0], [6.85, -0.2], [7.8, -0.3]]);
    left.shoulder.rotation.y = keys([[5.4, 0.1], [6.2, -1.5], [6.35, -1.5], [6.7, 0.7], [7.7, 0.1]]) - 0.06 * idle * Math.sin(this.time * 1.1);
    left.elbow.rotation.y = keys([[5.4, -0.2], [6.2, -0.1], [6.7, -0.8], [7.7, -0.2]]);
    left.shoulder.rotation.x = 0.12;
    right.shoulder.rotation.x = -0.12;
    this.yeti.ball.visible = t < 6.6 || t > 7.8;
    this.snowball.visible = t >= 6.6 && t < 6.9;
    // Let go: from the hand, to the penguin at the top of its flight
    if (passed(6.6) || this.releaseAt.lengthSq() === 0) {
      this.yeti.group.updateMatrixWorld(true);
      this.slap.worldToLocal(this.yeti.ball.getWorldPosition(this.releaseAt));
    }
    if (this.snowball.visible) {
      const k = s(6.6, 6.9);
      this.snowball.position.lerpVectors(this.releaseAt, APEX, k).z += 1.2 * k * (1 - k);
    }

    // The penguins: the one playing, the others waiting, the last one back from the sea
    // The bull's eye, and the penguin stuck in it head first, half in, the way it came
    const bull = new THREE.Vector3(this.iceberg.position.x - ICEBERG_SIZE / 2, 0, ICEBERG_SIZE * TARGET_HEIGHT);
    const way = bull.clone().sub(APEX).normalize();
    const target = bull.clone().addScaledVector(way, -0.5 * PENGUIN_SIZE);
    const dive = Math.atan2(way.x, way.z);
    const heading = (from: THREE.Vector3, to: THREE.Vector3) => Math.atan2(to.y - from.y, to.x - from.x);
    // A jump (or a dive) from a point to another, up by height, turning from one pitch to another
    const leap = (p: THREE.Group, from: THREE.Vector3, to: THREE.Vector3, u: number, height: number, pitch: number[]) => {
      p.position.lerpVectors(from, to, u).z += height * 4 * u * (1 - u);
      p.rotation.set(0, pitch[0] + (pitch[1] - pitch[0]) * u, heading(from, to));
      flap(p, 0.2 + 1.2 * Math.sin(Math.PI * u));
    };
    // The one playing first (k = 0): the others watch it, and cheer when it's hit and when it lands
    const order = this.penguins.map((p, i) => ({ p, i, k: (((i - round) % PENGUINS) + PENGUINS) % PENGUINS })).sort((a, b) => a.k - b.k);
    const watched = new THREE.Vector3();
    const cheer = Math.max(0, 1 - Math.abs(t - 7.3) / 0.6, 1 - Math.abs(t - 8.4) / 0.7);
    const LIE = Math.PI / 2 - 0.25; // on the belly, the head a bit up
    for (const { p, i, k } of order) {
      const place = (round + k) % SPOTS.length;
      const spot = SPOTS[place].clone().setZ(top);
      const lying = LYING.includes(place);
      p.visible = true;
      p.scale.setScalar(PENGUIN_SIZE);
      p.rotation.set(0, 0, 0);
      flap(p, 0.15);
      for (const f of p.userData.feet as THREE.Group[]) f.rotation.y = 0;
      const waddle = (from: THREE.Vector3, to: THREE.Vector3, u: number) => {
        p.position.lerpVectors(from, to, u);
        p.position.z = top + (u > 0 && u < 1 ? 0.12 * Math.abs(Math.sin(this.time * 12)) : 0);
        p.rotation.z = heading(from, to);
        if (u > 0 && u < 1) p.rotation.x = 0.18 * Math.sin(this.time * 12);
      };
      if (k === PENGUINS - 1 && t < 1.6) {
        // Last round's: under the sea, then popping out of it onto the island, from the far side
        p.visible = t >= 1.0;
        leap(p, POP_FROM.clone().setZ(-0.6), spot, s(1.0, 1.6), 1.5, [0, 0]);
        continue;
      }
      if (k > 0) {
        // Watching the game, turning to follow the one playing; jumping and flapping when cheering
        const jump = cheer * Math.abs(Math.sin(this.time * 9 + i * 1.3));
        p.position.copy(spot);
        p.rotation.z = heading(spot, watched) + 0.15 * Math.sin(this.time * 0.5 + i * 1.7);
        flap(p, 0.15 + cheer * (0.6 + 0.6 * Math.sin(this.time * 16 + i)));
        if (lying) {
          // Lying down once back from the sea, rolling about when cheering
          const down = k === PENGUINS - 1 ? ease(s(1.6, 2.2)) : 1;
          p.rotation.y = LIE * down;
          p.rotation.x = 0.3 * cheer * Math.sin(this.time * 12 + i);
          p.position.z += 0.22 * PENGUIN_SIZE * down + 0.1 * jump;
        } else {
          p.position.z += 0.35 * jump;
        }
        continue;
      }
      if (t < 0.8) {
        // Getting up if lying, off to the sea
        waddle(spot, DIVE_FROM.clone().setZ(top), ease(s(lying ? 0.25 : 0, 0.8)));
        if (lying) {
          const down = 1 - ease(s(0, 0.25));
          p.rotation.y = LIE * down;
          p.position.z += 0.22 * PENGUIN_SIZE * down;
        }
      } else if (t < 1.2) {
        // Into the sea, head first
        leap(p, DIVE_FROM.clone().setZ(top), DIVE_TO, s(0.8, 1.2), 0.6, [0, Math.PI / 2]);
      } else if (t < 3.4) {
        // Swimming across, on its belly, half out of the water
        p.position.lerpVectors(DIVE_TO, SWIM_TO, s(1.2, 3.4)).z = 0.03 * Math.sin(this.time * 6);
        p.rotation.set(0, Math.PI / 2, heading(DIVE_TO, SWIM_TO));
        flap(p, 0.4 + 0.5 * Math.sin(this.time * 10));
      } else if (t < 3.8) {
        // Out onto the island of the yeti
        leap(p, SWIM_TO, HOP_TO.clone().setZ(top), s(3.4, 3.8), 1.0, [Math.PI / 2, 0]);
      } else if (t < 4.3) {
        waddle(HOP_TO, EDGE, ease(s(3.8, 4.3)));
      } else if (t < 4.9) {
        // At the edge: turning to the orca, a pause, then getting ready, crouching and leaning forward
        const ready = ease(s(4.55, 4.9));
        p.position.copy(EDGE).setZ(top);
        p.rotation.z = heading(HOP_TO, EDGE) * (1 - ease(s(4.3, 4.45)));
        p.rotation.y = 0.35 * ready;
        p.scale.z = PENGUIN_SIZE * (1 - 0.22 * ready + 0.04 * ready * Math.sin(this.time * 18));
        flap(p, 0.15 + 0.9 * ready);
      } else if (t < 5.4) {
        // The jump onto the tail of the orca
        leap(p, EDGE.clone().setZ(top), onTail, s(4.9, 5.4), 2.5, [0.35, 0]);
        p.rotation.z = 0;
      } else if (t < 5.85) {
        p.position.copy(onTail);
        p.rotation.y = -tail;
      } else if (t < 6.9) {
        // Up in the air, turning over slowly
        const u = s(5.85, 6.9);
        p.position.set(APEX.x, 0, onTail.z + (APEX.z - onTail.z) * (1 - (1 - u) ** 2));
        p.rotation.y = -Math.PI * 0.8 * u;
        flap(p, 0.6 + 0.6 * Math.sin(this.time * 20));
      } else if (t < 7.9) {
        // Hit: shot like a rocket, straight to the iceberg, head first, flippers along its body, a
        // trail of snow behind
        p.position.lerpVectors(APEX, target, s(6.9, 7.9));
        p.rotation.y = dive;
        flap(p, -0.05);
        this.puffs.emit(p.position, way.clone().negate().setZ(0).normalize(), 2, 0.08);
      } else {
        // Stuck in the bull's eye, the feet kicking; then out, and down into the sea, gone under
        const fall = Math.max(0, t - 11);
        p.position.copy(target).addScaledVector(way, -0.3 * fall).z -= 3 * fall * fall;
        p.rotation.y = dive - 2 * fall;
        p.rotation.x = fall ? 0 : 0.08 * Math.sin(this.time * 9);
        flap(p, fall ? 1 : -0.05);
        (p.userData.feet as THREE.Group[]).forEach((f, j) => (f.rotation.y = 0.7 * Math.sin(this.time * 16 + j * Math.PI)));
        p.visible = p.position.z > -0.3;
      }
      watched.copy(p.visible ? p.position : target);
    }
    // Snow flying off the penguin where the snowball hits it, and on the iceberg
    if (passed(6.9)) this.puffs.emit(APEX, new THREE.Vector3(1, 0, 0), 120, 0.35);
    if (passed(7.9)) this.puffs.emit(bull, new THREE.Vector3(-1, 0, 0), 300, 0.45);
    this.slapBefore = t;
  }

  /** Whether something moving is close enough to the camera (km) to be worth animating. */
  near(camera: THREE.Camera): boolean {
    if (!this.group.visible) return false;
    if (this.falling.length || this.sliding.length) return true;
    return [this.train, this.table.group, this.whirlpool, this.slap].some((o) => this.isPlaced(o) && camera.position.distanceTo(o.position) < 600);
  }

  /** Where an egg is (km, with the exaggeration), and where to see it from; null if unknown or not drawn. */
  whereIs(id: string): { target: THREE.Vector3; position: THREE.Vector3 } | null {
    const egg = this.eggs.find((e) => e.id === id);
    if (!egg || !this.isPlaced(egg.objects[0])) return null;
    const target = egg.objects[0].position.clone();
    // The orca slap: halfway to the iceberg, a bit up, to see the whole flight
    if (egg.objects[0] === this.slap) target.copy(this.slap.localToWorld(new THREE.Vector3(this.iceberg.position.x / 2, 0, 3)));
    // The fossil is put on the bed in the shader
    if (egg.objects[0] === this.fossil) target.z = ((this.bedAt(target.x, target.y) ?? 0) * this.exaggeration) / 1000;
    // The whirlpool: its mouth at the sea surface, not its point on the seafloor
    if (egg.objects[0] === this.whirlpool) target.z = 0;
    // From the side of the home view, but the vessel from downhill, the side of the mountain it crashed into
    const side = new THREE.Vector3(0, -1, 0);
    if (egg.objects[0] === this.vessel) {
      const h = (x: number, y: number) => this.heightAt(x, y) ?? 0;
      const [x, y, r] = [target.x, target.y, 8];
      const downhill = new THREE.Vector3(h(x - r, y) - h(x + r, y), h(x, y - r) - h(x, y + r), 0);
      if (downhill.lengthSq() > 0) side.copy(downhill.normalize());
    }
    // The orca slap from across the flight
    if (egg.objects[0] === this.slap) side.set(Math.sin(this.slap.rotation.z), -Math.cos(this.slap.rotation.z), 0);
    // Steep enough over a deep trough (the fossil), not to have its walls in the way
    const [view, elevation] = egg.view;
    const e = THREE.MathUtils.degToRad(elevation);
    // Further if the terrain around (the bed for the fossil) would be in the way: the camera
    // well above the highest ground within the view and on the way to it
    const inTrough = egg.objects[0] === this.fossil;
    const groundAt = inTrough ? this.bedAt : this.heightAt;
    let highest = -Infinity;
    // All around, for the walls of the trough of the fossil
    for (let k = 0; k < (inTrough ? 24 : 0); k++) {
      const a = (k / 24) * Math.PI * 2;
      for (const r of [5, 15, 30]) highest = Math.max(highest, groundAt(target.x + r * Math.cos(a), target.y + r * Math.sin(a)) ?? -Infinity);
    }
    for (let t = 0.1; t <= 1; t += 0.1) {
      const d = t * view * Math.cos(e);
      highest = Math.max(highest, groundAt(target.x + d * side.x, target.y + d * side.y) ?? -Infinity);
    }
    const above = (highest * this.exaggeration) / 1000 - target.z + 10;
    const distance = Math.max(view, above / Math.sin(e));
    const position = target.clone().add(new THREE.Vector3(side.x * Math.cos(e), side.y * Math.cos(e), Math.sin(e)).multiplyScalar(distance));
    return { target, position };
  }

  private isPlaced(object: THREE.Object3D): boolean {
    return this.placed.get(object) ?? true;
  }

  /**
   * Draw only what is close enough to be seen (km from the camera), so the many small parts of
   * the eggs cost nothing from afar. Before each frame.
   */
  cull(camera: THREE.Camera): void {
    const near = (o: THREE.Object3D, km: number) => this.isPlaced(o) && camera.position.distanceTo(o.position) < km;
    this.vessel.visible = near(this.vessel, 900);
    this.fossil.visible = near(this.fossil, 900);
    this.whirlpool.visible = near(this.whirlpool, 4000);
    this.cromlech.visible = near(this.cromlech, 600);
    this.station.visible = this.island.visible = this.table.group.visible = near(this.station, 700);
    this.train.visible = near(this.train, 1500);
    this.slap.visible = near(this.slap, 900);
  }

  /** Knock a standing core over, away from the middle of the ring or across it, at random. */
  private topple(core: THREE.Mesh): void {
    core.userData.broken = core.userData.core;
    delete core.userData.core;
    this.falling.push({ core, toward: Math.random() * Math.PI * 2, angle: 0.03, speed: 0 });
  }

  /**
   * The cores falling over from their foot, faster and faster (like an upturned pendulum), then,
   * once on the snow, broken into two or three pieces that slide apart a little.
   */
  private moveCores(dt: number): void {
    this.falling = this.falling.filter((f) => {
      const { core, toward } = f;
      f.speed += 25 * Math.sin(f.angle) * dt;
      f.angle = Math.min(Math.PI / 2, f.angle + f.speed * dt);
      const axis = new THREE.Vector3(-Math.sin(toward), Math.cos(toward), 0);
      core.quaternion.setFromAxisAngle(axis, f.angle);
      if (f.angle < Math.PI / 2) return true;
      // On the snow: pieces in its place, lying along the fall
      const { radius, piece, ice } = core.userData.broken as { radius: number; piece: THREE.BufferGeometry; ice: THREE.Material };
      core.visible = false;
      const length = core.scale.z;
      const n = 2 + Math.floor(2 * Math.random());
      const dir = new THREE.Vector3(Math.cos(toward), Math.sin(toward), 0);
      const across = new THREE.Vector3(-dir.y, dir.x, 0);
      let along = 0;
      for (let k = 0; k < n; k++) {
        const l = k === n - 1 ? length - along : (length - along) * (0.35 + 0.3 * Math.random());
        const m = mesh(piece, ice);
        m.scale.y = l;
        m.rotation.z = toward - Math.PI / 2 + 0.3 * (Math.random() - 0.5);
        const from = core.position.clone().addScaledVector(dir, along + l / 2).setZ(radius);
        const to = from.clone().addScaledVector(dir, 0.006 * k + 0.004 * Math.random()).addScaledVector(across, 0.01 * (Math.random() - 0.5));
        m.position.copy(from);
        this.cromlech.add(m);
        this.sliding.push({ piece: m, from, to, t: 0 });
        along += l;
      }
      return false;
    });
    this.sliding = this.sliding.filter((s) => {
      s.t = Math.min(1, s.t + dt / 0.4);
      s.piece.position.lerpVectors(s.from, s.to, 1 - (1 - s.t) ** 3);
      return s.t < 1;
    });
  }

  get names(): { id: string; title: string }[] {
    return this.eggs.map(({ id, title }) => ({ id, title }));
  }

  /** The thing under a point of the canvas (px), or null. Only close to it: not found by chance, from afar. */
  pick(camera: THREE.Camera, x: number, y: number, width: number, height: number): Egg | null {
    if (!this.group.visible) return null;
    this.raycaster.setFromCamera(new THREE.Vector2((x / width) * 2 - 1, 1 - (y / height) * 2), camera);
    for (const egg of this.eggs) {
      if (camera.position.distanceTo(egg.objects[0].position) > egg.reach) continue;
      if (egg.objects[0] === this.fossil) {
        if (this.overFossil(camera, x, y, width, height)) return egg;
      } else {
        const hits = this.raycaster.intersectObjects(egg.objects, true);
        if (!hits.length) continue;
        // A core of the cromlech under the mouse falls over
        const core = hits.find((h) => h.object.userData.core)?.object;
        if (core instanceof THREE.Mesh) this.topple(core);
        return egg;
      }
    }
    return null;
  }

  /**
   * Whether a pixel is over the fossil, which the raycaster can't see (it is
   * moved onto the bed on the GPU): within its radius on screen around its
   * centre on the bed.
   */
  private overFossil(camera: THREE.Camera, x: number, y: number, width: number, height: number): boolean {
    if (!this.isPlaced(this.fossil)) return false;
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
