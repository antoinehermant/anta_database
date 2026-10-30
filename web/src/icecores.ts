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

export interface IceCore {
  name: string; // label of the flag
  site: string;
  lat: number;
  lon: number;
  elevation: number; // m above sea level, of the surface at the drill site
  depth: number; // m, drilled depth (to the bed for most)
  bottomAge: string | null; // oldest dated ice, as published
  drilled: string; // field seasons
  reference: string;
  poleScale?: number; // taller pole, so the flag clears a neighbour's
}

export const ICE_CORES: IceCore[] = [
  { name: "EDC", site: "EPICA Dome C", lat: -75.1017, lon: 123.3482, elevation: 3233, depth: 3270, bottomAge: "≈ 800 ka", drilled: "1996–2004", reference: "EPICA community members (2004), Nature" },
  { name: "BELDC", site: "Beyond EPICA, Little Dome C", lat: -75.2992, lon: 122.4451, elevation: 3230, depth: 2800, bottomAge: "> 1.2 Ma", drilled: "2021–2025", reference: "Beyond EPICA – Oldest Ice (2025); Lilien et al. (2021), The Cryosphere", poleScale: 2.2 },
  { name: "Dome Fuji", site: "Dome Fuji (DF2)", lat: -77.3167, lon: 39.7, elevation: 3810, depth: 3035, bottomAge: "≈ 720 ka", drilled: "2003–2007", reference: "Dome Fuji Ice Core Project Members (2017), Science Advances" },
  { name: "Vostok", site: "Vostok (5G)", lat: -78.4644, lon: 106.8372, elevation: 3488, depth: 3769, bottomAge: "≈ 420 ka (meteoric ice)", drilled: "1990–2012", reference: "Petit et al. (1999), Nature; Vasiliev et al. (2012)" },
  { name: "EDML", site: "EPICA Dronning Maud Land (Kohnen)", lat: -75.0025, lon: 0.0684, elevation: 2892, depth: 2774, bottomAge: "≈ 150 ka (undisturbed record)", drilled: "2001–2006", reference: "EPICA community members (2006), Nature" },
  { name: "TALDICE", site: "Talos Dome", lat: -72.8167, lon: 159.1833, elevation: 2315, depth: 1620, bottomAge: "≈ 250 ka", drilled: "2004–2008", reference: "Stenni et al. (2011), Nature Geoscience; Buiron et al. (2011), Climate of the Past" },
  { name: "Taylor Dome", site: "Taylor Dome", lat: -77.7964, lon: 158.7239, elevation: 2365, depth: 554, bottomAge: "≈ 130 ka", drilled: "1993–1994", reference: "Steig et al. (2000), Geografiska Annaler" },
  { name: "Law Dome", site: "Law Dome (DSS)", lat: -66.7725, lon: 112.8113, elevation: 1370, depth: 1196, bottomAge: null, drilled: "1988–1993", reference: "Morgan et al. (1997), Journal of Glaciology" },
  { name: "SPICEcore", site: "South Pole", lat: -89.99, lon: -98.16, elevation: 2835, depth: 1751, bottomAge: "≈ 54 ka", drilled: "2014–2016", reference: "Winski et al. (2019), Climate of the Past" },
  { name: "WDC", site: "WAIS Divide (WDC06A)", lat: -79.467, lon: -112.085, elevation: 1766, depth: 3405, bottomAge: "≈ 68 ka", drilled: "2006–2011", reference: "WAIS Divide Project Members (2013), Nature" },
  { name: "Byrd", site: "Byrd Station", lat: -80.0167, lon: -119.5167, elevation: 1530, depth: 2164, bottomAge: null, drilled: "1966–1968", reference: "Ueda and Garfield (1969), CRREL" },
  { name: "Siple Dome", site: "Siple Dome A", lat: -81.65, lon: -148.81, elevation: 621, depth: 1004, bottomAge: "≈ 100 ka", drilled: "1996–1999", reference: "Taylor et al. (2004), Journal of Glaciology" },
  { name: "Skytrain", site: "Skytrain Ice Rise", lat: -79.741, lon: -78.5448, elevation: 784, depth: 651, bottomAge: "≈ 130 ka", drilled: "2018–2019", reference: "Mulvaney et al. (2023), Climate of the Past" },
];

// Other sites, not shown: add them back to ICE_CORES to show them
export const OTHER_ICE_CORES: IceCore[] = [
  { name: "ABN", site: "Aurora Basin North", lat: -71.17, lon: 111.37, elevation: 2690, depth: 303, bottomAge: null, drilled: "2013–2014", reference: "Servettaz et al. (2023), Climate of the Past" },
  { name: "RICE", site: "Roosevelt Island (RICE)", lat: -79.36, lon: -161.71, elevation: 550, depth: 763, bottomAge: "≈ 83 ka", drilled: "2011–2013", reference: "Lee et al. (2020), Climate of the Past" },
  { name: "Fletcher", site: "Fletcher Promontory", lat: -77.9, lon: -82.61, elevation: 873, depth: 654, bottomAge: null, drilled: "2011–2012", reference: "Mulvaney et al. (2014), Annals of Glaciology" },
  { name: "Berkner", site: "Berkner Island (Thyssenhöhe)", lat: -79.55, lon: -45.68, elevation: 890, depth: 948, bottomAge: null, drilled: "2002–2005", reference: "Mulvaney et al. (2007), Annals of Glaciology" },
  { name: "JRI", site: "James Ross Island (Mount Haddington)", lat: -64.2017, lon: -57.685, elevation: 1542, depth: 363, bottomAge: null, drilled: "2008", reference: "Mulvaney et al. (2012), Nature" },
];

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
const CORE_PX = 10;
const POLE_PX = 1.5;
const CORE_RADIUS = 3;
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

interface Site {
  core: IceCore;
  x: number;
  y: number;
  column: THREE.Mesh; // the ice core, below the surface
  pole: THREE.Mesh;
  flag: THREE.Sprite;
  flagAspect: number;
}

const CORE_COLOR = new THREE.Color("#e6f4fc");
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
      const column = new THREE.Mesh(cylinder, new THREE.MeshLambertMaterial({ color: CORE_COLOR, emissive: "#2c4f66" }));
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
