import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import Stats from "three/examples/jsm/libs/stats.module.js";
import GUI from "lil-gui";
import {
  colormapTexture,
  gridGeometry,
  gridOf,
  loadHeightTexture,
  loadMaskTexture,
  loadMeta,
  setLayerTextures,
  sharedUniforms,
  terrainMaterial,
  type TerrainMeta,
} from "./terrain";
import { ageGradientCss, Isochrones, type Feature } from "./irh";

const DATA_URL = `${import.meta.env.BASE_URL}data/terrain`;
const IRH_URL = `${import.meta.env.BASE_URL}data/irh`;
// Resolutions (km) of the quick preview, then of the final mesh and shading texture
const PREVIEW = { mesh: "8km", texture: "8km" };
const DETAIL = { mesh: "4km", texture: "2km" };

const BED_COLORS: [number, string][] = [
  [-7000, "#071a3d"],
  [-4000, "#0d3a73"],
  [-1500, "#2a6aa8"],
  [-300, "#6aa3d0"],
  [0, "#a7cfe4"],
  [1, "#5f7d4e"],
  [600, "#8c8a5a"],
  [1500, "#a6835c"],
  [2500, "#c4ae93"],
  [4000, "#f3efe8"],
];

// Renderer, scene, camera
const container = document.getElementById("app")!;
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color("#0b1320");

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 50000);
camera.up.set(0, 0, 1);
const HOME = { position: new THREE.Vector3(0, -5200, 6200), target: new THREE.Vector3(0, 0, 0) };

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.zoomToCursor = true;
controls.screenSpacePanning = false; // pan along the ground
controls.minDistance = 20;
controls.maxDistance = 15000;
controls.maxPolarAngle = 0.49 * Math.PI; // never below the horizon

function resetView(): void {
  camera.position.copy(HOME.position);
  controls.target.copy(HOME.target);
  controls.update();
  history.replaceState(null, "", window.location.pathname);
  requestRender();
}

// The view is kept in the URL (#view=camera x,y,z,target x,y,z in km), so it can be shared
function viewFromUrl(): boolean {
  const match = window.location.hash.match(/view=([-\d.,]+)/);
  const v = match?.[1].split(",").map(Number);
  if (!v || v.length !== 6 || v.some(Number.isNaN)) return false;
  camera.position.set(v[0], v[1], v[2]);
  controls.target.set(v[3], v[4], v[5]);
  controls.update();
  return true;
}

function viewToUrl(): void {
  const v = [...camera.position.toArray(), ...controls.target.toArray()].map((x) => x.toFixed(0));
  history.replaceState(null, "", `#view=${v.join(",")}`);
}
controls.addEventListener("end", viewToUrl);

// Render on demand: only when the view or a setting changed
let needsRender = true;
function requestRender(): void {
  needsRender = true;
}
// Hover waits until the view has been still for a moment
let lastViewChange = 0;
controls.addEventListener("change", () => {
  lastViewChange = performance.now();
  isochrones.invalidate();
  pointerMoved = true;
  requestRender();
});
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  isochrones.setResolution(window.innerWidth, window.innerHeight);
  isochrones.invalidate();
  requestRender();
});

const stats = new Stats();
if (import.meta.env.DEV) document.body.appendChild(stats.dom);

renderer.setAnimationLoop(() => {
  controls.update();
  updateHover();
  if (!needsRender) return;
  needsRender = false;
  renderer.render(scene, camera);
  stats.update();
});

// Terrain layers
const shared = sharedUniforms();
const bed = new THREE.Mesh(
  new THREE.BufferGeometry(),
  terrainMaterial(shared, { colormap: colormapTexture(BED_COLORS) }),
);
const surface = new THREE.Mesh(
  new THREE.BufferGeometry(),
  terrainMaterial(shared, {
    color: new THREE.Color("#f4f8fc"),
    shelfColor: new THREE.Color("#bcd8ee"),
    opacity: 0.5,
  }),
);
for (const mesh of [bed, surface]) {
  mesh.frustumCulled = false; // positions are displaced in the shader
  mesh.visible = false;
  scene.add(mesh);
}
// The transparent surface is drawn after everything opaque underneath it
surface.renderOrder = 1;

// Isochrones
const isochrones = new Isochrones();
isochrones.setResolution(window.innerWidth, window.innerHeight);
isochrones.setExaggeration(shared.exaggeration.value);
scene.add(isochrones.group);

// Hover: find the line under the cursor at most once per frame, not while dragging
const tooltip = document.getElementById("tooltip")!;
let pointer: { x: number; y: number } | null = null;
let pointerMoved = false;
let hovered: number | null = null;

renderer.domElement.addEventListener("pointermove", (event) => {
  pointer = event.buttons ? null : { x: event.offsetX, y: event.offsetY };
  pointerMoved = true;
});
renderer.domElement.addEventListener("pointerleave", () => {
  pointer = null;
  pointerMoved = true;
});

function formatAge(f: Feature): string {
  if (f.age_ka === null) return "not assigned";
  return f.age_unc_ka === null ? `${f.age_ka} ka` : `${f.age_ka} ± ${f.age_unc_ka} ka`;
}

function showTooltip(f: Feature, x: number, y: number): void {
  const rows: [string, string][] = [
    ["Age", formatAge(f)],
    ["Flight ID", f.flight_id],
    ["IRH", String(f.irh)],
  ];
  tooltip.replaceChildren();
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = f.citation ?? f.dataset;
  tooltip.append(title);
  for (const [key, value] of rows) {
    const row = document.createElement("div");
    const k = document.createElement("span");
    k.className = "key";
    k.textContent = `${key}: `;
    row.append(k, value);
    tooltip.append(row);
  }
  if (f.surface !== "SURF_ELEV") {
    const note = document.createElement("div");
    note.className = "note";
    note.textContent = `Elevation from depth and the ${f.surface} surface`;
    tooltip.append(note);
  }
  // Keep the tooltip inside the window
  const left = Math.min(x + 16, window.innerWidth - tooltip.offsetWidth - 8);
  const top = y + 16 + tooltip.offsetHeight > window.innerHeight ? y - tooltip.offsetHeight - 12 : y + 16;
  tooltip.style.transform = `translate(${left}px, ${top}px)`;
  tooltip.style.visibility = "visible";
}

function updateHover(): void {
  if (!pointerMoved) return;
  const moving = performance.now() - lastViewChange < 150;
  if (moving && pointer) {
    tooltip.style.visibility = "hidden";
    return;
  }
  pointerMoved = false;
  const canvas = renderer.domElement;
  const feature = pointer
    ? isochrones.pick(camera, pointer.x, pointer.y, canvas.clientWidth, canvas.clientHeight)
    : null;
  if (feature === null || !pointer) {
    tooltip.style.visibility = "hidden";
  } else {
    showTooltip(isochrones.features[feature], pointer.x, pointer.y);
  }
  if (feature !== hovered) {
    hovered = feature;
    isochrones.setHighlight(feature);
    renderer.domElement.style.cursor = feature === null ? "" : "pointer";
    requestRender();
  }
}

let maskTexture: THREE.DataTexture | null = null;

async function loadLevel(meta: TerrainMeta, res: { mesh: string; texture: string }): Promise<void> {
  const level = meta.resolutions[res.texture];
  const [bedTex, surfaceTex, mask] = await Promise.all([
    loadHeightTexture(DATA_URL, level, "bed", meta.nodata),
    loadHeightTexture(DATA_URL, level, "surface", meta.nodata),
    loadMaskTexture(DATA_URL, level),
  ]);
  const grid = gridOf(level);
  setLayerTextures(bed.material, bedTex, mask, grid);
  setLayerTextures(surface.material, surfaceTex, mask, grid);
  maskTexture?.dispose();
  maskTexture = mask;

  const geometry = gridGeometry(gridOf(meta.resolutions[res.mesh]));
  bed.geometry.dispose();
  bed.geometry = geometry;
  surface.geometry = geometry;
  bed.visible = settings.bed;
  surface.visible = settings.surface;
  requestRender();
}

// Settings panel
const settings = {
  exaggeration: shared.exaggeration.value as number,
  bed: true,
  surface: true,
  surfaceOpacity: 0.5,
  isochrones: true,
  lineWidth: 2.5,
  lightAzimuth: 315, // degrees clockwise from grid north (+y)
  lightElevation: 40,
  resetView,
};

function updateLight(): void {
  const az = THREE.MathUtils.degToRad(settings.lightAzimuth);
  const el = THREE.MathUtils.degToRad(settings.lightElevation);
  shared.lightDir.value.set(Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el));
  requestRender();
}
updateLight();

const gui = new GUI({ title: "AntADatabase" });
gui
  .add(settings, "exaggeration", 1, 100, 1)
  .name("Vertical exaggeration")
  .onChange((v: number) => {
    shared.exaggeration.value = v;
    isochrones.setExaggeration(v);
    pointerMoved = true;
    requestRender();
  });
const layers = gui.addFolder("Layers");
layers
  .add(settings, "bed")
  .name("Bed (Bedmap3)")
  .onChange((v: boolean) => {
    bed.visible = v;
    requestRender();
  });
layers
  .add(settings, "surface")
  .name("Ice surface")
  .onChange((v: boolean) => {
    surface.visible = v;
    requestRender();
  });
layers
  .add(settings, "surfaceOpacity", 0.05, 1, 0.05)
  .name("Surface opacity")
  .onChange((v: number) => {
    surface.material.uniforms.opacity.value = v;
    requestRender();
  });
layers
  .add(settings, "isochrones")
  .name("Isochrones")
  .onChange((v: boolean) => {
    isochrones.group.visible = v;
    pointerMoved = true;
    requestRender();
  });
layers
  .add(settings, "lineWidth", 1, 8, 0.5)
  .name("Line width")
  .onChange((v: number) => {
    isochrones.setLineWidth(v);
    requestRender();
  });
const light = gui.addFolder("Light");
light.add(settings, "lightAzimuth", 0, 360, 1).name("Azimuth").onChange(updateLight);
light.add(settings, "lightElevation", 5, 90, 1).name("Elevation").onChange(updateLight);
light.close();
gui.add(settings, "resetView").name("Reset view");

// Start: quick preview, then the detailed terrain
const loading = document.getElementById("loading")!;
if (!viewFromUrl()) resetView();
loadMeta(DATA_URL)
  .then(async (meta) => {
    document.getElementById("bedmap-doi")?.setAttribute("href", meta.doi);
    await loadLevel(meta, PREVIEW);
    loading.textContent = "Loading detailed terrain…";
    await loadLevel(meta, DETAIL);
    loading.remove();
  })
  .catch((error) => {
    loading.textContent = String(error.message ?? error);
    console.error(error);
  });

isochrones
  .load(IRH_URL)
  .then(() => {
    const legend = document.getElementById("legend")!;
    legend.querySelector<HTMLElement>(".bar")!.style.background = ageGradientCss();
    legend.querySelector(".max")!.textContent = `${isochrones.maxAgeKa} ka`;
    legend.style.visibility = "visible";
    requestRender();
  })
  .catch((error) => console.error(error));
