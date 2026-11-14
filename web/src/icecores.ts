/**
 * Deep ice core sites: a core from the bottom of the borehole up to the ice
 * surface, then a pole with a flag above it, so the sites stand out.
 *
 * Positions, surface elevations and drilled depths are from the publications
 * of each project (see `reference`). Sizes on screen stay readable at any
 * zoom: cores and poles are at least a few pixels wide, and flags are sprites
 * of a fixed size in pixels.
 */
import * as THREE from "three";

export { ICE_CORES, OTHER_ICE_CORES, type IceCore } from "./icecore-sites";
import { ICE_CORES, type IceCore } from "./icecore-sites";

/** EPSG:3031 (WGS 84 Antarctic polar stereographic, true scale at 71° S) in km. */
export function polarStereographic(lat: number, lon: number): [number, number] {
  const a = 6378137;
  const e = 0.0818191908426215;
  const tsOf = (phi: number) => Math.tan(Math.PI / 4 - phi / 2) / ((1 - e * Math.sin(phi)) / (1 + e * Math.sin(phi))) ** (e / 2);
  // South pole aspect: the formulas of the north pole with the signs flipped
  const phi = -THREE.MathUtils.degToRad(lat);
  const phiC = THREE.MathUtils.degToRad(71);
  const mC = Math.cos(phiC) / Math.sqrt(1 - (e * Math.sin(phiC)) ** 2);
  const rho = (a * mC * tsOf(phi)) / tsOf(phiC);
  const lambda = THREE.MathUtils.degToRad(lon);
  return [(rho * Math.sin(lambda)) / 1000, (rho * Math.cos(lambda)) / 1000];
}

// Minimum widths on screen (px), and sizes in the scene (km) when zoomed in
const CORE_PX = 7;
const POLE_PX = 1.5;
const CORE_RADIUS = 2;
const POLE_RADIUS = 0.3;
// Pole above the surface: in exaggerated metres, but at least this many pixels
const POLE_HEIGHT_M = 1200;
const POLE_PX_HEIGHT = 40;
const FLAG_PX = 22; // height of the flags

function flagTexture(label: string): { texture: THREE.CanvasTexture; aspect: number } {
  const scale = 3; // sharp on high density screens
  const height = FLAG_PX * scale;
  const font = `600 ${13 * scale}px system-ui, sans-serif`;
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d")!;
  context.font = font;
  const notch = 5 * scale;
  const width = Math.ceil(context.measureText(label).width + 10 * scale + notch);
  canvas.width = width;
  canvas.height = height;
  context.font = font;
  // Flag with a notched end
  context.fillStyle = "#d8352a";
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
  context.fillText(label, 5 * scale, height / 2 + scale);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return { texture, aspect: width / height };
}

/**
 * The look of a core along its depth (top of the canvas: the surface): clear ice, bubbly
 * near the top (the bubbles turn into clathrates deeper), a few dust and ash layers, and at
 * the bottom the ice mixed with rock, where there is some. Half transparent, the mud not.
 */
function coreTexture(core: IceCore): THREE.CanvasTexture {
  const [width, height] = [96, 1024];
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d")!;
  let seed = [...core.name].reduce((h, c) => (h * 31 + c.charCodeAt(0)) % 2147483647, 7) || 1;
  const random = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
  // Ice: pale blue, bluer deeper
  const ice = context.createLinearGradient(0, 0, 0, height);
  ice.addColorStop(0, "rgba(206, 232, 250, 0.72)");
  ice.addColorStop(1, "rgba(150, 198, 236, 0.8)");
  context.fillStyle = ice;
  context.fillRect(0, 0, width, height);
  // A few layers: dust (grey, wide and faint) and ash (darker, thinner), wavy and blurred,
  // fading in and out across the core
  context.filter = "blur(1.5px)";
  const layers = 4 + Math.floor(3 * random());
  for (let k = 0; k < layers; k++) {
    const y = height * (0.05 + 0.8 * random());
    const ash = random() < 0.35;
    const thick = ash ? 1.5 + random() : 3 + 4 * random();
    const [phase, wave, tilt] = [2 * Math.PI * random(), 1 + 2 * random(), 4 * (random() - 0.5)];
    for (let x = 0; x < width; x += 2) {
      const alpha = (ash ? 0.5 : 0.25) * (0.6 + 0.4 * Math.sin((x / width) * 2 * Math.PI + phase));
      context.fillStyle = ash ? `rgba(70, 72, 80, ${alpha.toFixed(2)})` : `rgba(140, 140, 128, ${alpha.toFixed(2)})`;
      const dy = wave * Math.sin((x / width) * 4 * Math.PI + phase) + (tilt * x) / width;
      context.fillRect(x, y + dy, 2, thick);
    }
  }
  context.filter = "none";
  // Bubbles, fewer with depth, gone below about 40 % (the bubbles turn into clathrates):
  // white with a darker rim, wider than tall (the canvas is stretched along the core)
  for (let k = 0; k < 900; k++) {
    const y = height * 0.45 * random();
    if (random() > 1 - y / (0.45 * height)) continue;
    const r = 1 + 2 * random();
    const [x, alpha] = [width * random(), 0.6 + 0.35 * random()];
    context.beginPath();
    context.ellipse(x, y, r, 0.6 * r, 0, 0, 2 * Math.PI);
    context.fillStyle = `rgba(255, 255, 255, ${alpha.toFixed(2)})`;
    context.fill();
    context.strokeStyle = `rgba(90, 130, 170, ${(0.5 * alpha).toFixed(2)})`;
    context.lineWidth = 0.6;
    context.stroke();
  }
  // Mud at the bottom, with a ragged top and grains of rock
  if (core.basalDebris) {
    const top = height * (1 - core.basalDebris / core.depth);
    const mud = context.createLinearGradient(0, top - 12, 0, height);
    mud.addColorStop(0, "rgba(120, 100, 76, 0)");
    mud.addColorStop(0.25, "rgba(112, 92, 68, 0.85)");
    mud.addColorStop(1, "rgba(84, 68, 52, 1)");
    context.fillStyle = mud;
    context.beginPath();
    context.moveTo(0, height);
    for (let x = 0; x <= width; x += 4) context.lineTo(x, top - 10 + 8 * Math.sin(x * 0.2 + 3 * random()));
    context.lineTo(width, height);
    context.closePath();
    context.fill();
    for (let k = 0; k < 500; k++) {
      const y = top + (height - top) * Math.sqrt(random());
      const grey = Math.floor(40 + 120 * random());
      context.fillStyle = `rgba(${grey + 20}, ${grey + 8}, ${grey}, 1)`;
      context.fillRect(width * random(), y, 1 + 2 * random(), 1 + random());
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

interface Site {
  core: IceCore;
  x: number;
  y: number;
  column: THREE.Mesh; // the ice core, below the surface
  pole: THREE.Mesh;
  flag: THREE.Sprite;
  flagAspect: number;
}

const CORE_COLOR = new THREE.Color("#ffffff"); // the colours are in the texture
const CORE_HIGHLIGHT = new THREE.Color("#ffd166");

export class IceCores {
  readonly group = new THREE.Group();
  // Flags are drawn last, over everything (lines and the transparent surface
  // included), so they are never hidden. The order of a group applies to all
  // its children, so they have their own group.
  private flags = new THREE.Group();
  private sites: Site[] = [];
  private exaggeration = 1;
  private raycaster = new THREE.Raycaster();
  private highlighted: Site | null = null;

  constructor(cores: IceCore[] = ICE_CORES) {
    // Cylinders of unit radius and height, standing on z = 0
    const cylinder = new THREE.CylinderGeometry(1, 1, 1, 24, 1).rotateX(Math.PI / 2).translate(0, 0, 0.5);
    const poleMaterial = new THREE.MeshLambertMaterial({ color: "#3a3f47" });
    this.flags.renderOrder = 100;
    this.group.add(this.flags);
    for (const core of cores) {
      const [x, y] = polarStereographic(core.lat, core.lon);
      // Half transparent like ice (the alpha of the texture), the mud at the bottom not
      const column = new THREE.Mesh(
        cylinder,
        new THREE.MeshLambertMaterial({ color: CORE_COLOR, emissive: "#1c3442", map: coreTexture(core), transparent: true }),
      );
      const pole = new THREE.Mesh(cylinder, poleMaterial);
      const { texture, aspect } = flagTexture(core.name);
      // Opaque: the notched end is cut out (alphaTest), not blended
      const flag = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: texture, sizeAttenuation: false, depthTest: false, depthWrite: false, alphaTest: 0.5, toneMapped: false }),
      );
      flag.center.set(0, 0); // the bottom left corner is at the top of the pole
      const site: Site = { core, x, y, column, pole, flag, flagAspect: aspect };
      for (const object of [column, pole, flag]) {
        object.userData.site = site;
        object.frustumCulled = false;
      }
      this.group.add(column, pole);
      this.flags.add(flag);
      this.sites.push(site);
    }
    // Lit like the terrain: from above, with a soft sky light
    this.group.add(new THREE.HemisphereLight("#ffffff", "#8090a0", 2.2));
    const sun = new THREE.DirectionalLight("#ffffff", 1.2);
    sun.position.set(-1, 1, 2);
    this.group.add(sun);
  }

  get cores(): IceCore[] {
    return this.sites.map((s) => s.core);
  }

  setExaggeration(exaggeration: number): void {
    this.exaggeration = exaggeration;
  }

  /** Sizes for the current view: called before each render. */
  update(camera: THREE.PerspectiveCamera, height: number): void {
    const z = (m: number) => (m * this.exaggeration) / 1000;
    const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const position = new THREE.Vector3();
    for (const site of this.sites) {
      const { core, column, pole, flag, flagAspect } = site;
      const top = z(core.elevation);
      // Size of a pixel (km) at the site
      const pixel = (2 * tan * camera.position.distanceTo(position.set(site.x, site.y, top))) / height;
      const coreRadius = Math.max(CORE_RADIUS, (CORE_PX / 2) * pixel);
      column.position.set(site.x, site.y, z(core.elevation - core.depth));
      column.scale.set(coreRadius, coreRadius, Math.max(1e-3, z(core.depth)));
      const poleRadius = Math.max(POLE_RADIUS, (POLE_PX / 2) * pixel);
      const poleHeight = Math.max(z(POLE_HEIGHT_M), POLE_PX_HEIGHT * pixel) * (core.poleScale ?? 1);
      pole.position.set(site.x, site.y, top);
      pole.scale.set(poleRadius, poleRadius, poleHeight);
      flag.position.set(site.x, site.y, top + poleHeight);
      // Sprites without size attenuation are sized in clip space (2 = screen height)
      const flagHeight = (2 * FLAG_PX) / height;
      flag.scale.set(flagHeight * flagAspect * tan, flagHeight * tan, 1);
    }
  }

  /** Ice core under a point of the canvas (px), or null. */
  pick(camera: THREE.Camera, x: number, y: number, width: number, height: number): IceCore | null {
    if (!this.group.visible) return null;
    this.raycaster.setFromCamera(new THREE.Vector2((x / width) * 2 - 1, 1 - (y / height) * 2), camera);
    // Flags first: they are drawn on top
    const flags = this.raycaster.intersectObjects(this.sites.map((s) => s.flag), false);
    const hits = flags.length ? flags : this.raycaster.intersectObjects(this.sites.flatMap((s) => [s.column, s.pole]), false);
    return (hits[0]?.object.userData.site as Site | undefined)?.core ?? null;
  }

  setHighlight(core: IceCore | null): void {
    const site = this.sites.find((s) => s.core === core) ?? null;
    if (site === this.highlighted) return;
    if (this.highlighted) (this.highlighted.column.material as THREE.MeshLambertMaterial).color.copy(CORE_COLOR);
    if (site) (site.column.material as THREE.MeshLambertMaterial).color.copy(CORE_HIGHLIGHT);
    this.highlighted = site;
  }
}
