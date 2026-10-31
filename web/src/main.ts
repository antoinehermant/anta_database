import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import Stats from "three/examples/jsm/libs/stats.module.js";
import GUI from "lil-gui";
import {
  colormapTexture,
  gridGeometry,
  gridOf,
  gridPositions,
  loadHeightTexture,
  loadMaskTexture,
  loadMeta,
  setLayerTextures,
  sharedUniforms,
  terrainMaterial,
  verticesNearData,
  type Grid,
  type TerrainMeta,
} from "./terrain";
import { ageKey, BASAL_UNIT_COLOR, COLOR_BY, isBasalUnit, Isochrones, loadIndex, Tracks, type AgeKey, type ColorBy, type IrhIndex, type Feature, type Track } from "./irh";
import { closer, type Hit, type LineLayer } from "./lines";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { PivotField } from "./pivot";
import { BED_STOPS, SPEED, ticks } from "./colormaps";
import { Legend } from "./legend";
import { KeyNavigation } from "./keys";
import { EasterEggs } from "./eggs";
import { onThemeChange, themeButton } from "./theme";
import { IceCores, type IceCore } from "./icecores";
import { Aircraft, flightLines, type FlightLine } from "./aircraft";
import { FlowField, FlowLines, loadSpeedTexture, loadVelocityMeta, LOG_SPEED_RANGE } from "./velocity";
import { SUBMISSION, startPreview } from "./preview";
import { Tutorial, tutorialSeen } from "./tutorial";
import { CONFIDENCE_STYLES, GROUNDING_LINE_COLORS, GroundingLines, IceDivides, type HeightAt, type Scenario } from "./outlines";
import { ProfilePanel, type ProfilePoint, type Tooltip } from "./profile";

// Data exported by anta_database.web: served with the app in development
// (web/public/data), or from the bucket when VITE_DATA_URL is set at build time
const DATA_ROOT = (import.meta.env.VITE_DATA_URL || `${import.meta.env.BASE_URL}data`).replace(/\/$/, "");
const DATA_URL = `${DATA_ROOT}/terrain`;
// The isochrones of the database, or of a submission in its private preview (?preview=<id>&key=<key>)
const IRH_URL = SUBMISSION ? SUBMISSION.url : `${DATA_ROOT}/irh`;
const VELOCITY_URL = `${DATA_ROOT}/velocity`;
const OUTLINES_URL = `${DATA_ROOT}/outlines`;
// Resolution of the quick preview, then of the shading texture
const PREVIEW = "8km";
const DETAIL = "2km";
// Mesh resolution of each quality setting; the shading is always from the texture
const QUALITY: Record<string, string> = { Fast: "8km", Detailed: "4km" };

// Renderer, scene, camera
const container = document.getElementById("app")!;
const renderer = new THREE.WebGLRenderer({ antialias: true });
// Full resolution when still, lower while the view moves
const FULL_PIXEL_RATIO = Math.min(window.devicePixelRatio, 2);
const MOVING_PIXEL_RATIO = Math.max(0.75, FULL_PIXEL_RATIO / 2);
renderer.setPixelRatio(FULL_PIXEL_RATIO);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color();
// Light or dark theme (dark by default): from the button in the corner, saved in the browser
function applyTheme(): void {
  // Same background as the page, from the stylesheet
  const background = getComputedStyle(document.documentElement).getPropertyValue("--background").trim();
  (scene.background as THREE.Color).set(background || "#1c2633");
  requestRender();
}
onThemeChange(applyTheme);
themeButton(document.getElementById("theme-toggle") as HTMLButtonElement);

const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 1, 50000);
camera.up.set(0, 0, 1);
// Home view: the camera relative to the rotation centre (the target), tilted by
// 47°, and the centre a bit towards the viewer from the middle of the continent,
// so the continent sits higher in the frame and fills more of it
const HOME_OFFSET = new THREE.Vector3(0, -4205, 4500);
const HOME_CENTRE = new THREE.Vector2(0, -400);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.zoomToCursor = true;
controls.screenSpacePanning = false; // pan along the ground
controls.minDistance = 20;
controls.maxDistance = 15000;
controls.maxPolarAngle = 0.49 * Math.PI; // never below the horizon of the rotation centre

// Dragging rotates; with Ctrl (or Shift, Cmd) held, or the right button, it moves over the
// ground (OrbitControls). Pointing at an edge of the view also moves towards it (below)

// Rotate, zoom and pan around the middle of the ice under the centre of the
// screen, updated when an interaction starts. The centre moves along the line
// of sight, so the view doesn't jump.
let pivotField: PivotField | null = null;
// Heights of the terrain as drawn, once the first (coarse) level is loaded
let terrainReady: (heights: HeightAt) => void;
const terrainHeights = new Promise<HeightAt>((resolve) => (terrainReady = resolve));
controls.addEventListener("start", () => {
  if (!pivotField) return;
  const direction = camera.getWorldDirection(new THREE.Vector3());
  const hit = pivotField.intersect(camera.position, direction, shared.exaggeration.value, 2 * controls.maxDistance);
  if (hit && hit.distanceTo(camera.position) >= controls.minDistance) controls.target.copy(hit);
});

/** Back to the whole continent: flying there, or at once (when the viewer starts). */
function resetView(instant = false): void {
  forgetZoom();
  flight = null;
  const { x, y } = HOME_CENTRE;
  const centreZ = pivotField ? (pivotField.heightAt(x, y) * shared.exaggeration.value) / 1000 : 0;
  const target = new THREE.Vector3(x, y, centreZ);
  const position = target.clone().add(HOME_OFFSET);
  // The home view has no #view= in the address
  const clearUrl = () => history.replaceState(null, "", window.location.pathname + window.location.search);
  if (!instant) {
    flyTo(target, position, 1200, clearUrl, false, true);
    return;
  }
  controls.target.copy(target);
  camera.position.copy(position);
  controls.update();
  clearUrl();
  requestRender();
}

// Extent of the dataset of a private preview, looked at once known (and again with the terrain heights)
let submissionBox: number[] | null = null;

/** Look at an area (EPSG:3031, m: xmin, xmax, ymin, ymax) from the direction of the home view. */
function frameBox([xmin, xmax, ymin, ymax]: number[]): void {
  const x = (xmin + xmax) / 2000;
  const y = (ymin + ymax) / 2000;
  const size = Math.max(xmax - xmin, ymax - ymin) / 1000;
  const z = pivotField ? (pivotField.heightAt(x, y) * shared.exaggeration.value) / 1000 : 0;
  controls.target.set(x, y, z);
  const distance = THREE.MathUtils.clamp(4 * size, 200, controls.maxDistance);
  camera.position.copy(controls.target).add(HOME_OFFSET.clone().setLength(distance));
  controls.update();
  requestRender();
}

// Clicking a line flies to it, and a "Reset view" button pops up (gone once the view is reset,
// from it or the menu). Closing the stratigraphy returns to the view before the first of these
// flights (not to the line before)
let viewBeforeZoom: { target: THREE.Vector3; position: THREE.Vector3 } | null = null;
const backButton = document.createElement("button");
backButton.id = "back-view";
backButton.type = "button";
backButton.textContent = "↺ Reset view";
backButton.hidden = true;
backButton.addEventListener("click", () => resetView());
/** Fly back to the view before the first flight to a line, if any. */
function goBack(): void {
  if (viewBeforeZoom) flyTo(viewBeforeZoom.target, viewBeforeZoom.position);
  forgetZoom();
}
document.body.append(backButton);
function forgetZoom(): void {
  viewBeforeZoom = null;
  backButton.hidden = true;
}

// A flight of the view: moved a bit each frame by the animation loop (stepFlight()), at the
// lower resolution used while moving, and stopped by any drag
let flight: {
  target0: THREE.Vector3;
  position0: THREE.Vector3;
  target: THREE.Vector3;
  position: THREE.Vector3;
  start: number;
  ms: number;
  onEnd: () => void;
  orbit: boolean;
  animate: boolean;
} | null = null;

/**
 * Move the view smoothly to a rotation centre and camera position, so you see where you go;
 * onEnd once there (by default, the view goes in the address). With orbit, the camera turns
 * around the centre (its distance, tilt and direction change evenly) instead of going straight.
 * With animate, the animations go on meanwhile (far views, where they are cheap).
 */
function flyTo(target: THREE.Vector3, position: THREE.Vector3, ms = 800, onEnd = viewToUrl, orbit = false, animate = false): void {
  flight = { target0: controls.target.clone(), position0: camera.position.clone(), target, position, start: performance.now(), ms, onEnd, orbit, animate };
  requestRender();
}

function stepFlight(now: number): void {
  if (!flight) return;
  const t = Math.min(1, (now - flight.start) / flight.ms);
  const k = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2; // ease in and out
  controls.target.lerpVectors(flight.target0, flight.target, k);
  if (flight.orbit) {
    // Camera around the centre, z up: distance, angle from the vertical, direction
    const toSpherical = (v: THREE.Vector3) => [v.length(), Math.acos(THREE.MathUtils.clamp(v.z / v.length(), -1, 1)), Math.atan2(v.y, v.x)];
    const [r0, phi0, theta0] = toSpherical(flight.position0.clone().sub(flight.target0));
    const [r1, phi1, theta1] = toSpherical(flight.position.clone().sub(flight.target));
    const turn = Math.atan2(Math.sin(theta1 - theta0), Math.cos(theta1 - theta0)); // the shorter way
    const [r, phi, theta] = [r0 + (r1 - r0) * k, phi0 + (phi1 - phi0) * k, theta0 + turn * k];
    camera.position.set(Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi)).multiplyScalar(r).add(controls.target);
  } else {
    camera.position.lerpVectors(flight.position0, flight.position, k);
  }
  if (t === 1) {
    const { onEnd } = flight;
    flight = null;
    onEnd();
  }
}
controls.addEventListener("start", () => (flight = null));

// Closest the view of a clicked line comes (km from the camera to the middle of the line), so a
// short line isn't seen too close: higher zooms in less
const LINE_VIEW_MIN_DISTANCE = 1000;
const AUTO_ZOOM_KEY = "auto-zoom";

/**
 * Show a whole line (points in km, NaN between its parts): seen from the side, across its main
 * direction and as tilted as the view already is, so the line spans the screen.
 */
function zoomToLine(x: Float32Array, y: Float32Array): void {
  let [xmin, xmax, ymin, ymax] = [Infinity, -Infinity, Infinity, -Infinity];
  let first = -1;
  let last = -1;
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) continue;
    if (first < 0) first = i;
    last = i;
    [xmin, xmax, ymin, ymax] = [Math.min(xmin, x[i]), Math.max(xmax, x[i]), Math.min(ymin, y[i]), Math.max(ymax, y[i])];
  }
  if (first < 0) return;
  viewBeforeZoom ??= { target: controls.target.clone(), position: camera.position.clone() };
  backButton.hidden = false;
  const [cx, cy] = [(xmin + xmax) / 2, (ymin + ymax) / 2];
  const length = Math.max(Math.hypot(xmax - xmin, ymax - ymin), 20);
  // Across the line from its first to its last point (or the box, for a line coming back)
  let [dx, dy] = [x[last] - x[first], y[last] - y[first]];
  if (Math.hypot(dx, dy) < 0.3 * length) [dx, dy] = xmax - xmin >= ymax - ymin ? [1, 0] : [0, 1];
  const norm = Math.hypot(dx, dy);
  let [nx, ny] = [-dy / norm, dx / norm];
  // From the side nearer the current camera, so the view turns as little as possible
  if (nx * (camera.position.x - cx) + ny * (camera.position.y - cy) < 0) [nx, ny] = [-nx, -ny];
  const z = pivotField ? (pivotField.heightAt(cx, cy) * shared.exaggeration.value) / 1000 : 0;
  const target = new THREE.Vector3(cx, cy, z);
  // Far enough for the length of the line to fit the width of the screen
  const halfWidth = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.aspect;
  const distance = THREE.MathUtils.clamp((0.65 * length) / halfWidth, LINE_VIEW_MIN_DISTANCE, controls.maxDistance);
  // The tilt the view already has (its angle above the horizontal)
  const current = camera.position.clone().sub(controls.target);
  const tilt = Math.asin(THREE.MathUtils.clamp(current.z / current.length(), -1, 1));
  const offset = new THREE.Vector3(nx * Math.cos(tilt), ny * Math.cos(tilt), Math.sin(tilt)).multiplyScalar(distance);
  flyTo(target, target.clone().add(offset));
}

// Momentum after a drag: the view glides on, but stops once it barely moves
// (under MOMENTUM_STOP_PX per frame) or after MOMENTUM_MS at most, so the full
// resolution and the hover come back quickly
const MOMENTUM_MS = 1500;
const MOMENTUM_STOP_PX = 0.1;
let interacting = false;
let releasedAt = 0;
const previousCamera = new THREE.Vector3();
controls.addEventListener("start", () => {
  interacting = true;
});
controls.addEventListener("end", () => {
  interacting = false;
  releasedAt = performance.now();
});

/** Size of a pixel (km) at the rotation centre. */
function pixelAtTarget(): number {
  const distance = camera.position.distanceTo(controls.target);
  return (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * distance) / renderer.domElement.clientHeight;
}

function stopMomentum(): void {
  if (interacting || !releasedAt) return;
  const moved = camera.position.distanceTo(previousCamera) / pixelAtTarget();
  if (moved < MOMENTUM_STOP_PX || performance.now() - releasedAt > MOMENTUM_MS) {
    // Remaining rotation and pan of the damping (fields of OrbitControls)
    const internal = controls as unknown as { _sphericalDelta: THREE.Spherical; _panOffset: THREE.Vector3 };
    internal._sphericalDelta.set(0, 0, 0);
    internal._panOffset.set(0, 0, 0);
    releasedAt = 0;
  }
}

// The view is kept in the URL (#view=camera x,y,z,target x,y,z in km), so it can be shared
/**
 * Look straight down, grid north up, like a map: at the middle of the
 * continent when the whole continent is in view (from about the home view
 * distance), else at the centre of the view.
 */
function topView(): void {
  const offset = camera.position.clone().sub(controls.target);
  const distance = offset.length();
  const target = controls.target.clone();
  if (distance >= 0.8 * HOME_OFFSET.length()) {
    const centreZ = pivotField ? (pivotField.heightAt(0, 0) * shared.exaggeration.value) / 1000 : 0;
    target.set(0, 0, centreZ);
  }
  // A hair south of the vertical, so grid north is up; turning and rising around the centre
  const lean = 1e-3;
  const position = target.clone().add(new THREE.Vector3(0, -Math.sin(lean), Math.cos(lean)).multiplyScalar(distance));
  flyTo(target, position, 1000, viewToUrl, true, true);
}

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
  history.replaceState(null, "", `${window.location.search}#view=${v.join(",")}`);
}
controls.addEventListener("end", viewToUrl);
// W A S D, the arrow keys and the pad in the bottom right corner move the view too
const keyNavigation = new KeyNavigation(camera, controls, viewToUrl, resetView);
keyNavigation.addPad(document.body);

// Pointing at an edge of the view moves over the ground towards it: after a moment there, so
// passing by on the way to the menu or a button doesn't move, then faster closer to the edge.
// Not while dragging, nor over the panels (the pointer is then not on the canvas)
const EDGE_KEY = "edge-move";
const EDGE_PX = 180; // width of the edges that move the view, in pixels
const EDGE_DELAY_MS = 200;
const EDGE_RAMP_MS = 3000;
// The left edge while the menu is open, next to it: the pointer crosses it on the way to the
// menu, so it moves nothing (0) or only from a thinner band and after a longer moment
const EDGE_MENU_PX = 50;
const EDGE_MENU_DELAY_MS = 800;
let edgePointer: { x: number; y: number } | null = null;
let edgeSince = 0; // when the pointer came near an edge, 0 when it isn't

function updateEdgeMove(now: number): void {
  let [along, across] = [0, 0];
  let delay = EDGE_DELAY_MS;
  if (edgePointer && settings.edgeMove && !flight) {
    const { x, y } = edgePointer;
    // The view starts right of the menu when it is open
    const menuOpen = !sidebar.classList.contains("closed");
    const left = menuOpen ? sidebar.getBoundingClientRect().right : 0;
    const near = (distance: number, width = EDGE_PX) => (width > 0 ? THREE.MathUtils.clamp(1 - distance / width, 0, 1) : 0);
    const nearLeft = near(x - left, menuOpen ? EDGE_MENU_PX : EDGE_PX);
    along = near(y) - near(window.innerHeight - y);
    across = near(window.innerWidth - x) - nearLeft;
    if (menuOpen && nearLeft) delay = Math.max(EDGE_DELAY_MS, EDGE_MENU_DELAY_MS);
  }
  if (!along && !across) {
    edgeSince = 0;
  } else if (!edgeSince) {
    edgeSince = now;
  }
  const ramp = edgeSince ? THREE.MathUtils.clamp((now - edgeSince - delay) / EDGE_RAMP_MS, 0, 1) : 0;
  keyNavigation.setEdge(along * ramp, across * ramp);
}
renderer.domElement.addEventListener("pointermove", (event) => {
  edgePointer = event.buttons ? null : { x: event.clientX, y: event.clientY };
});
for (const type of ["pointerleave", "pointerdown"]) renderer.domElement.addEventListener(type, () => (edgePointer = null));
// The mouse left the browser window (to another window), or the window is no longer in front
document.addEventListener("mouseout", (event) => {
  if (!event.relatedTarget) edgePointer = null;
});
window.addEventListener("blur", () => (edgePointer = null));
document.addEventListener("visibilitychange", () => (edgePointer = null));
controls.addEventListener("end", () => {
  // Still flow lines are spread again over the new view
  if (flowLines && !settings.animateFlow) {
    updateFlowArea();
    flowLines.respawn();
    requestRender();
  }
});

// Render on demand: only when the view or a setting changed
let needsRender = true;
function requestRender(): void {
  needsRender = true;
}
// Hover waits until the view has been still for a moment, and only runs
// again when the mouse moves
let lastViewChange = 0;
controls.addEventListener("change", () => {
  lastViewChange = performance.now();
  for (const layer of lineLayers) layer.invalidate();
  clearHover();
  requestRender();
});
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  for (const layer of lineLayers) {
    layer.setResolution(window.innerWidth, window.innerHeight);
    layer.invalidate();
  }
  flowLines?.setResolution(window.innerWidth, window.innerHeight);
  requestRender();
});

const stats = new Stats();
if (import.meta.env.DEV) {
  // Bottom right, above the move pad (the profile window opens top right)
  Object.assign(stats.dom.style, { left: "auto", right: "10px", top: "auto", bottom: "130px" });
  stats.dom.id = "stats";
  document.body.appendChild(stats.dom);
}

// Flow lines move at most 30 times per second, to spare the GPU
const FLOW_INTERVAL = 1000 / 30;
let lastFlowUpdate = 0;

let lastAircraftUpdate = 0;

// Animations (aircraft, flow lines) redraw the whole scene 30 times a second,
// which keeps the GPU busy. They pause while the window is in the background
// or nobody used the viewer for IDLE_MS, so the GPU is idle again, and go on
// with the next mouse move, touch or key.
const IDLE_MS = 2 * 60 * 1000;
let lastInput = performance.now();
for (const type of ["pointermove", "pointerdown", "wheel", "keydown"]) {
  window.addEventListener(type, () => (lastInput = performance.now()), { passive: true });
}
function animating(): boolean {
  return document.hasFocus() && performance.now() - lastInput < IDLE_MS;
}

let lastFrame = 0;
let lastRender = 0;
const GLIDE_FRAME_MS = 1000 / 30;
renderer.setAnimationLoop((time: number) => {
  stepFlight(performance.now());
  updateEdgeMove(performance.now());
  keyNavigation.update(Math.min(0.1, (time - lastFrame) / 1000));
  lastFrame = time;
  controls.update();
  // Moving by itself (a flight, keys, the pad or an edge): drawn at most 30 times a second, and
  // without the animations (but during the flights home and to the map), so a laptop keeps up
  // however long it lasts
  const gliding = flight !== null || keyNavigation.active;
  const animate = animating() && (flight ? flight.animate : !keyNavigation.active);
  stopMomentum();
  previousCamera.copy(camera.position);
  updateHover();
  if (!animate) {
    // Go on from where they stopped, without a jump
    lastAircraftUpdate = lastFlowUpdate = time;
  }
  if (animate && aircraft.group.visible && time - lastAircraftUpdate >= FLOW_INTERVAL) {
    const dt = Math.min(0.1, (time - lastAircraftUpdate) / 1000);
    aircraft.update(dt, camera, renderer.domElement.clientHeight);
    eggs.update(dt);
    lastAircraftUpdate = time;
    needsRender = true;
  }
  if (animate && flowLines?.mesh.visible && settings.animateFlow && time - lastFlowUpdate >= FLOW_INTERVAL) {
    updateFlowArea();
    flowLines.update(Math.min(0.1, (time - lastFlowUpdate) / 1000));
    lastFlowUpdate = time;
    needsRender = true;
  }
  const moving = performance.now() - lastViewChange < 200;
  const ratio = moving && settings.lowResWhileMoving ? MOVING_PIXEL_RATIO : FULL_PIXEL_RATIO;
  if (renderer.getPixelRatio() !== ratio) {
    renderer.setPixelRatio(ratio);
    needsRender = true;
  }
  if (!needsRender) return;
  if (gliding && time - lastRender < GLIDE_FRAME_MS) return;
  lastRender = time;
  needsRender = false;
  if (iceCores.group.visible) iceCores.update(camera, renderer.domElement.clientHeight);
  // Dashes of the tracks: the same length on screen at any zoom
  tracks.lines.setDashPixel(pixelAtTarget());
  if (groundingLines.group.visible) groundingLines.setDashPixel(pixelAtTarget());
  renderer.render(scene, camera);
  stats.update();
});

// Terrain layers
const shared = sharedUniforms();
const bed = new THREE.Mesh(
  new THREE.BufferGeometry(),
  // Colours are at fractions of the colour range, set with the range uniform
  terrainMaterial(shared, { colormap: colormapTexture(BED_STOPS) }),
);
const surface = new THREE.Mesh(
  new THREE.BufferGeometry(),
  terrainMaterial(shared, {
    color: new THREE.Color("#f4f8fc"),
    shelfColor: new THREE.Color("#bcd8ee"),
    opacity: 0.9,
  }),
);
for (const mesh of [bed, surface]) {
  mesh.frustumCulled = false; // positions are displaced in the shader
  mesh.visible = false;
  scene.add(mesh);
}
// The transparent surface is drawn after everything opaque underneath it,
// then the velocities, the tracks and the flow lines on top of it
surface.renderOrder = 1;

// Isochrones, and their radar tracks at the surface (loaded when first shown)
const isochrones = new Isochrones();
const tracks = new Tracks();
tracks.group.renderOrder = 3;
const lineLayers: LineLayer[] = [isochrones.lines, tracks.lines];
for (const layer of lineLayers) {
  layer.setResolution(window.innerWidth, window.innerHeight);
  layer.setExaggeration(shared.exaggeration.value);
  scene.add(layer.group);
}
// Grounding lines (RAISED and present) on the bed, and ice divides (IMBIE2 basins); off at start
const groundingLines = new GroundingLines();
const iceDivides = new IceDivides();
groundingLines.setVisible(false);
iceDivides.setVisible(false);
for (const layer of [...groundingLines.lineLayers, iceDivides.lines]) {
  layer.setResolution(window.innerWidth, window.innerHeight);
  layer.setExaggeration(shared.exaggeration.value);
  lineLayers.push(layer);
}
iceDivides.setExaggeration(shared.exaggeration.value);
scene.add(groundingLines.group, iceDivides.group);

const irhIndex = SUBMISSION
  ? startPreview(document.getElementById("preview")!).then((status) => {
      if (status.bbox && !window.location.hash) {
        submissionBox = status.bbox;
        frameBox(submissionBox);
      }
      import("./preview-download")
        .then(({ showDownload }) => showDownload(document.getElementById("preview-download")!, SUBMISSION!.url, status))
        .catch(console.error);
      return loadIndex(IRH_URL);
    })
  : loadIndex(IRH_URL);

// Deep ice core sites
const iceCores = new IceCores();
iceCores.setExaggeration(shared.exaggeration.value);
iceCores.group.visible = false; // off by default
scene.add(iceCores.group);

// Survey aircraft flying along the tracks, with their radar pulses
// One per dataset, flying the legs of its tracks, found once the tracks are loaded
const AIRCRAFT_DATASETS = ["Franke_2025", "Sanderson_2024", "Yan_2025", "Wang_2023", "Cavitte_2020", "Bodart_2021", "Bodart_2025b"];
let legs: Map<string, FlightLine[]> | null = null;
function aircraftLegs(dataset: string): FlightLine[] {
  if (!tracks.lines.loaded) return [];
  legs ??= flightLines(
    tracks.lines.featureCount,
    (f) => tracks.lines.featureSegments(f),
    (f) => tracks.tracks[f]?.dataset ?? "",
  );
  return legs.get(dataset) ?? [];
}
const aircraft = new Aircraft(
  AIRCRAFT_DATASETS,
  aircraftLegs,
  (f) => tracks.lines.isHidden(f),
  (x, y) => pivotField?.bedAt(x, y) ?? null,
);
aircraft.setExaggeration(shared.exaggeration.value);
scene.add(aircraft.group);

// Two hidden things to find
const ground = (x: number, y: number): number | null => {
  const [top, bed] = [pivotField?.surfaceAt(x, y) ?? null, pivotField?.bedAt(x, y) ?? null];
  return top === null ? bed : bed === null ? top : Math.max(top, bed);
};
// The grid of the bed mesh, for the fossil to lie on it
const bedMesh = { meshOrigin: { value: new THREE.Vector2() }, meshDx: { value: 8 } };
const u = bed.material.uniforms;
const eggs = new EasterEggs(
  ground,
  (x, y) => pivotField?.bedAt(x, y) ?? null,
  (x, y) => pivotField?.surfaceAt(x, y) ?? null,
  { heightTex: u.heightTex, texOrigin: u.texOrigin, texDx: u.texDx, texSize: u.texSize, exaggeration: shared.exaggeration, ...bedMesh },
);
eggs.setExaggeration(shared.exaggeration.value);
scene.add(eggs.group);

// Ice velocity (loaded when first shown): the ice surface coloured by speed, and flow lines
let velocity: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial> | null = null;
let flowLines: FlowLines | null = null;
let velocityLoading: Promise<void> | null = null;

function loadVelocity(): Promise<void> {
  velocityLoading ??= (async () => {
    const meta = await loadVelocityMeta(VELOCITY_URL);
    const [speed, field] = await Promise.all([
      loadSpeedTexture(VELOCITY_URL, meta),
      FlowField.load(VELOCITY_URL, meta),
    ]);
    const colormap = colormapTexture(SPEED.stops);
    const material = terrainMaterial(shared, {
      colormap: { texture: colormap.texture, range: LOG_SPEED_RANGE },
      opacity: settings.velocityOpacity,
      velocity: { texture: speed, grid: gridOf(meta.grids.speed) },
    });
    // Same heights as the ice surface, updated with it
    const u = surface.material.uniforms;
    for (const name of ["heightTex", "maskTex", "texOrigin", "texDx", "texSize"]) material.uniforms[name] = u[name];
    velocity = new THREE.Mesh(surface.geometry, material);
    velocity.frustumCulled = false;
    velocity.renderOrder = 2;
    scene.add(velocity);

    flowLines = new FlowLines(field, u);
    flowLines.mesh.renderOrder = 4;
    scene.add(flowLines.mesh);
    updateFlowArea();
    flowLines.respawn();
    document.getElementById("velocity-doi")?.setAttribute("href", meta.doi);
  })();
  return velocityLoading;
}

/** Flow lines live around the centre of the view, over about the visible width. */
function updateFlowArea(): void {
  if (!flowLines) return;
  const halfHeight = camera.position.distanceTo(controls.target) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  flowLines.setArea(new THREE.Vector2(controls.target.x, controls.target.y), 1.2 * halfHeight * Math.max(1, camera.aspect));
}

function showVelocity(): void {
  const on = settings.velocity;
  if (on && !velocity) loadVelocity().then(showVelocity).catch(showError);
  if (velocity) velocity.visible = on;
  if (flowLines) flowLines.mesh.visible = on && settings.flowLines;
  document.getElementById("velocity-attribution")!.hidden = !on;
  velocityOptions.show(on);
  if (on && velocity) {
    velocityLegend.show("Ice speed (m/a)", SPEED, ticks({ min: 1, max: 10 ** LOG_SPEED_RANGE.y, log: true }));
  } else velocityLegend.hide();
  requestRender();
}

// Colour bars
const legends = document.getElementById("legends")!;
const isochroneLegend = new Legend(legends);
const velocityLegend = new Legend(legends);
const groundingLineLegend = new Legend(legends);

function showIsochroneLegend(): void {
  if (!settings.isochrones || !isochrones.lines.loaded) return isochroneLegend.hide();
  const scale = isochrones.colorScale();
  isochroneLegend.show(scale.title, scale.colormap, scale.ticks, isochrones.showsBasalUnit() ? [["Basal unit", BASAL_UNIT_COLOR]] : []);
}

function showError(error: Error): void {
  console.error(error);
  const message = document.createElement("div");
  message.className = "error";
  message.textContent = String(error.message ?? error);
  document.body.append(message);
  setTimeout(() => message.remove(), 8000);
}

// Hover: find the line under the cursor at most once per frame, not while dragging
const tooltip = document.getElementById("tooltip")!;
let pointer: { x: number; y: number } | null = null;
let pointerMoved = false;
let hovered: { layer: LineLayer; feature: number } | null = null;
let hoveredCore: IceCore | null = null;

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

function isochroneTooltip(f: Feature, hit: Hit): void {
  const rows: [string, string][] = isBasalUnit(f) ? [["Layer", "Basal unit (no age)"]] : [["Age", formatAge(f)]];
  rows.push(["Elevation", `${Math.round(isochrones.lines.elevation(hit.segment, hit.end))} m`]);
  const depth = isochrones.depth(hit.segment, hit.end);
  if (depth !== null) rows.push(["Depth", `${depth} m`]);
  const fdepth = isochrones.fractionalDepth(hit.segment, hit.end);
  if (fdepth !== null) rows.push(["Fractional depth", fdepth.toFixed(2)]);
  rows.push(["Line ID", f.line_id]);
  if (!isBasalUnit(f)) rows.push(["IRH", String(f.irh)]);
  const notes: string[] = [];
  if (f.surface !== "SURF_ELEV") notes.push(`Elevation from depth and the ${f.surface} surface`);
  if (fdepth !== null && f.thickness === "Bedmap3") notes.push("Fractional depth with the Bedmap3 ice thickness");
  showTooltip(f.citation ?? f.dataset, rows, notes.length ? notes.join(". ") : null);
}

function trackTooltip(t: Track, hit: Hit): void {
  const p = tracks.point(hit.segment, hit.end);
  const { ages, noAge, basalUnit } = tracks.datasetAges.get(t.dataset) ?? { ages: [], noAge: 0, basalUnit: false };
  const list = ages.map((a) => String(Number(a.toFixed(2))));
  if (noAge) list.push(`${noAge} without age`);
  const rows: [string, string][] = [
    ["Line ID", t.line_id],
    ["Surface elevation", `${Math.round(p.surface)} m`],
    ["Traced IRHs (IRH_NUM)", String(Math.round(p.irhNum))],
    [`IRH ages of the dataset${ages.length ? " (ka)" : ""}`, list.join(", ") || "none"],
  ];
  if (basalUnit) rows.push(["Also in the dataset", "a basal unit layer"]);
  showTooltip(t.citation ?? t.dataset, rows, t.surface === "SURF_ELEV" ? null : `Surface elevation from ${t.surface}`);
}

function iceCoreTooltip(core: IceCore): void {
  const rows: [string, string][] = [
    ["Position", `${Math.abs(core.lat).toFixed(2)}° S, ${Math.abs(core.lon).toFixed(2)}° ${core.lon < 0 ? "W" : "E"}`],
    ["Surface elevation", `${core.elevation} m`],
    ["Drilled depth", `${core.depth} m`],
  ];
  if (core.bottomAge) rows.push(["Oldest ice", core.bottomAge]);
  rows.push(["Drilled", core.drilled]);
  showTooltip(`${core.site} ice core`, rows, core.reference);
}

function showTooltip(heading: string, rows: [string, string][], footnote: string | null): void {
  tooltip.replaceChildren();
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = heading;
  tooltip.append(title);
  for (const [key, value] of rows) {
    const row = document.createElement("div");
    const k = document.createElement("span");
    k.className = "key";
    k.textContent = `${key}: `;
    row.append(k, value);
    tooltip.append(row);
  }
  if (footnote) {
    const note = document.createElement("div");
    note.className = "note";
    note.textContent = footnote;
    tooltip.append(note);
  }
}

function showTip(tip: Tooltip): void {
  showTooltip(tip.title, tip.rows, tip.note);
}

function placeTooltip(x: number, y: number): void {
  // Keep the tooltip inside the window
  const left = Math.min(x + 16, window.innerWidth - tooltip.offsetWidth - 8);
  const top = y + 16 + tooltip.offsetHeight > window.innerHeight ? y - tooltip.offsetHeight - 12 : y + 16;
  tooltip.style.transform = `translate(${left}px, ${top}px)`;
  tooltip.style.visibility = "visible";
}

function clearHover(): void {
  tooltip.style.visibility = "hidden";
  if (hoveredCore) {
    iceCores.setHighlight(null);
    hoveredCore = null;
    renderer.domElement.style.cursor = "";
  }
  if (hovered === null) return;
  hovered.layer.setHighlight(null);
  hovered = null;
  renderer.domElement.style.cursor = "";
}

/**
 * Whether the isochrones are hidden under a nearly opaque ice surface: the
 * camera is above the surface, which is at least 80 % opaque. Then they can't
 * be hovered (from below the surface, they can).
 */
function isochronesUnderIce(): boolean {
  if (!surface.visible || settings.surfaceOpacity < 0.8 || !pivotField) return false;
  const { x, y, z } = camera.position;
  const top = pivotField.surfaceAt(x, y);
  return top === null || z > (top * shared.exaggeration.value) / 1000;
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
  // Ice cores first: few, and drawn over the lines
  const core = pointer ? iceCores.pick(camera, pointer.x, pointer.y, canvas.clientWidth, canvas.clientHeight) : null;
  if (core !== hoveredCore) {
    iceCores.setHighlight(core);
    hoveredCore = core;
    requestRender();
  }
  const egg = pointer && !core ? eggs.pick(camera, pointer.x, pointer.y, canvas.clientWidth, canvas.clientHeight) : null;
  // Closest line of all layers
  let best: { layer: LineLayer; hit: Hit } | null = null;
  if (pointer && !core && !egg) {
    for (const layer of lineLayers) {
      if (layer === isochrones.lines && isochronesUnderIce()) continue;
      const hit = layer.pick(camera, pointer.x, pointer.y, canvas.clientWidth, canvas.clientHeight);
      if (hit && (!best || closer(hit.distance, hit.depth, best.hit))) best = { layer, hit };
    }
  }
  if (core && pointer) {
    iceCoreTooltip(core);
    placeTooltip(pointer.x, pointer.y);
  } else if (egg && pointer) {
    showTooltip(egg.title, egg.rows, egg.note);
    placeTooltip(pointer.x, pointer.y);
  } else if (!best || !pointer) {
    tooltip.style.visibility = "hidden";
  } else {
    if (best.layer === tracks.lines) trackTooltip(tracks.tracks[best.hit.feature], best.hit);
    else if (best.layer === isochrones.lines) isochroneTooltip(isochrones.features[best.hit.feature], best.hit);
    else if (best.layer === iceDivides.lines) showTip(iceDivides.describe(best.hit));
    else showTip(groundingLines.describe(best.layer, best.hit));
    placeTooltip(pointer.x, pointer.y);
  }
  const next = best ? { layer: best.layer, feature: best.hit.feature } : null;
  if (next?.layer !== hovered?.layer || next?.feature !== hovered?.feature) {
    hovered?.layer.setHighlight(null);
    next?.layer.setHighlight(next.feature);
    hovered = next;
    requestRender();
  }
  renderer.domElement.style.cursor = next === null && core === null && egg === null ? "" : "pointer";
}

// Stratigraphy of a line: click an isochrone or a track to open its profile
const profileMarker = new THREE.Group();
const markerMaterial = new LineMaterial({ linewidth: 3, color: 0xff4d4d, depthTest: false, transparent: true });
markerMaterial.resolution.set(window.innerWidth, window.innerHeight);
window.addEventListener("resize", () => markerMaterial.resolution.set(window.innerWidth, window.innerHeight));
const markerGeometry = new LineSegmentsGeometry();
const markerLine = new LineSegments2(markerGeometry, markerMaterial);
markerLine.frustumCulled = false;
markerLine.renderOrder = 11;
profileMarker.add(markerLine);
profileMarker.visible = false;
profileMarker.scale.z = shared.exaggeration.value / 1000;
scene.add(profileMarker);

/** Mark a point of the profile in 3D: a vertical line through the ice, from the bed to the surface. */
function markProfilePoint(point: ProfilePoint | null): void {
  profileMarker.visible = point !== null;
  if (point) {
    const top = Number.isFinite(point.surface) ? point.surface : (pivotField?.surfaceAt(point.x, point.y) ?? 0);
    const bottom = Number.isFinite(point.bed) ? point.bed : (pivotField?.bedAt(point.x, point.y) ?? top - 1000);
    markerGeometry.setPositions([point.x, point.y, bottom, point.x, point.y, top + 300]);
  }
  requestRender();
}

const profile = new ProfilePanel({
  colorOf: (feature) => isochrones.ageColor(feature),
  describeLayer: (feature, depth, point) => {
    const f = isochrones.features[feature];
    const rows: [string, string][] = isBasalUnit(f) ? [["Layer", "Basal unit (no age)"]] : [["Age", formatAge(f)]];
    rows.push(["Elevation", `${Math.round(point.surface - depth)} m`], ["Depth", `${Math.round(depth)} m`]);
    if (point.thickness > 0) rows.push(["Fractional depth", (depth / point.thickness).toFixed(2)]);
    rows.push(["Line ID", f.line_id]);
    if (!isBasalUnit(f)) rows.push(["IRH", String(f.irh)]);
    return { title: f.citation ?? f.dataset, rows, note: f.surface !== "SURF_ELEV" ? `Elevation from depth and the ${f.surface} surface` : null };
  },
  showTooltip: (tip, x, y) => {
    showTip(tip);
    placeTooltip(x, y);
  },
  hideTooltip: () => (tooltip.style.visibility = "hidden"),
  onPoint: markProfilePoint,
  // Closing the stratigraphy also goes back to where you were before flying to its line
  onClose: goBack,
});

// Lines (tracks) by dataset and line ID, with their profiles
const linesById = new Map<string, Track>();
irhIndex.then((index) => {
  for (const t of index.tracks) linesById.set(`${t.dataset}\u0000${t.line_id}`, t);
});

/** Open the profile of the isochrone or track under a pixel of the canvas, if any. */
function openProfileAt(x: number, y: number): void {
  const canvas = renderer.domElement;
  let best: { layer: LineLayer; hit: Hit } | null = null;
  for (const layer of [isochrones.lines, tracks.lines]) {
    if (!layer.group.visible || (layer === isochrones.lines && isochronesUnderIce())) continue;
    const hit = layer.pick(camera, x, y, canvas.clientWidth, canvas.clientHeight);
    if (hit && (!best || closer(hit.distance, hit.depth, best.hit))) best = { layer, hit };
  }
  if (!best) return;
  const line = best.layer === tracks.lines ? tracks.tracks[best.hit.feature] : isochrones.features[best.hit.feature];
  const track = linesById.get(`${line.dataset}\u0000${line.line_id}`);
  if (!track) return;
  profile
    .open(track, IRH_URL)
    .then(() => {
      // Unless another line was clicked meanwhile
      if (settings.autoZoom && profile.line === track) {
        const { x, y } = profile.points();
        zoomToLine(x, y);
      }
    })
    .catch(showError);
}

// A click (not a drag) on a line opens its profile
let pressed: { x: number; y: number; time: number } | null = null;
renderer.domElement.addEventListener("pointerdown", (event) => {
  pressed = event.button === 0 ? { x: event.clientX, y: event.clientY, time: performance.now() } : null;
});
renderer.domElement.addEventListener("pointerup", (event) => {
  const click = pressed && Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y) < 5 && performance.now() - pressed.time < 500;
  pressed = null;
  if (click) openProfileAt(event.offsetX, event.offsetY);
});

let maskTexture: THREE.DataTexture | null = null;
// Grid and data coverage of the current height textures, to rebuild the meshes
let textureGrid: Grid | null = null;
let coverage: { bed: Uint8Array; surface: Uint8Array } | null = null;
let terrainMeta: TerrainMeta | null = null;

async function loadLevel(meta: TerrainMeta, res: string): Promise<void> {
  const level = meta.resolutions[res];
  const [bedTex, surfaceTex, mask] = await Promise.all([
    loadHeightTexture(DATA_URL, level, "bed", meta.nodata),
    loadHeightTexture(DATA_URL, level, "surface", meta.nodata),
    loadMaskTexture(DATA_URL, level),
  ]);
  const grid = gridOf(level);
  if (!pivotField) {
    pivotField = PivotField.fromLayers(grid, bedTex.userData as never, surfaceTex.userData as never);
    const field = pivotField;
    terrainReady((layer, x, y) => field.smoothAt(layer, x, y));
    eggs.place();
    if (!window.location.hash) {
      if (submissionBox) frameBox(submissionBox);
      else resetView(true);
    }
  }
  textureGrid = grid;
  coverage = { bed: bedTex.userData.valid, surface: surfaceTex.userData.valid };
  // Heights are only kept on the CPU for the first (coarse) level
  bedTex.userData = {};
  surfaceTex.userData = {};
  setLayerTextures(bed.material, bedTex, mask, grid);
  setLayerTextures(surface.material, surfaceTex, mask, grid);
  maskTexture?.dispose();
  maskTexture = mask;
  terrainMeta = meta;
  buildMeshes(res === PREVIEW ? PREVIEW : QUALITY[settings.quality]);
}

/** Bed and surface meshes at a resolution, without triangles where a layer has no data. */
function buildMeshes(res: string): void {
  if (!terrainMeta || !textureGrid || !coverage) return;
  const grid = gridOf(terrainMeta.resolutions[res]);
  bedMesh.meshOrigin.value.set(grid.x0, grid.y0);
  bedMesh.meshDx.value = grid.dx;
  const positions = gridPositions(grid);
  bed.geometry.dispose();
  surface.geometry.dispose();
  bed.geometry = gridGeometry(grid, positions, verticesNearData(grid, textureGrid, coverage.bed));
  surface.geometry = gridGeometry(grid, positions, verticesNearData(grid, textureGrid, coverage.surface));
  if (velocity) velocity.geometry = surface.geometry;
  bed.visible = settings.bed;
  surface.visible = settings.surface;
  requestRender();
}

// Settings panel
const settings = {
  exaggeration: shared.exaggeration.value as number,
  bed: true,
  bedRange: 7000, // m, colours from -bedRange to +bedRange
  surface: true,
  surfaceOpacity: 0.9,
  velocity: false,
  velocityOpacity: 0.8,
  flowLines: true,
  animateFlow: true,
  isochrones: true,
  colorBy: "Age" as ColorBy,
  tracks: true,
  iceCores: false,
  groundingLines: false,
  groundingLineScenario: "A" as Scenario,
  // Times shown (ka, 0 for the present), by checkbox
  groundingLineTimes: Object.fromEntries(GROUNDING_LINE_COLORS.map(([age]) => [String(age), true])) as Record<string, boolean>,
  iceDivides: false,
  aircraft: true,
  lineWidth: 2.5,
  lightAzimuth: 315, // degrees clockwise from grid north (+y)
  lightElevation: 40,
  quality: "Fast",
  lowResWhileMoving: true,
  resetView,
  topView,
  edgeMove: localStorage.getItem(EDGE_KEY) !== "off",
  autoZoom: localStorage.getItem(AUTO_ZOOM_KEY) !== "off",
};

// Private preview of a submission: its isochrones visible through the ice surface, and no
// aircraft, whose animation redraws the close view of the dataset 30 times a second
if (SUBMISSION) {
  settings.aircraft = false;
  settings.surfaceOpacity = 0.5;
  surface.material.uniforms.opacity.value = settings.surfaceOpacity;
}

// Link preview screenshots (public/og-image.jpg, 1200×630) in development: ?screenshot
// hides the panels; the current image is ?screenshot#view=150,-4257,3560,150,-600,150
if (import.meta.env.DEV && new URLSearchParams(window.location.search).has("screenshot")) {
  document.body.classList.add("clean");
  Object.assign(settings, { surfaceOpacity: 0.6, lineWidth: 3, exaggeration: 100 });
  isochrones.lines.setLineWidth(settings.lineWidth);
  surface.material.uniforms.opacity.value = settings.surfaceOpacity;
  shared.exaggeration.value = settings.exaggeration;
  for (const layer of lineLayers) layer.setExaggeration(settings.exaggeration);
  iceCores.setExaggeration(settings.exaggeration);
  aircraft.setExaggeration(settings.exaggeration);
  eggs.setExaggeration(settings.exaggeration);
}

function updateBedRange(): void {
  bed.material.uniforms.colormapRange.value.set(-settings.bedRange, settings.bedRange);
  requestRender();
}

function updateLight(): void {
  const az = THREE.MathUtils.degToRad(settings.lightAzimuth);
  const el = THREE.MathUtils.degToRad(settings.lightElevation);
  shared.lightDir.value.set(Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el));
  requestRender();
}
aircraft.group.visible = settings.aircraft;
if (settings.aircraft) irhIndex.then((index) => tracks.load(IRH_URL, index)).catch(showError);
updateLight();
updateBedRange();
applyTheme();

const gui = new GUI({ container: document.getElementById("gui")!, title: "Settings" });

// The sidebar slides to the side with its arrow tab; closed at start on small screens
const sidebar = document.getElementById("sidebar")!;
const sidebarToggle = document.getElementById("sidebar-toggle")!;
function setSidebar(open: boolean): void {
  sidebar.classList.toggle("closed", !open);
  const label = open ? "Hide the menu" : "Show the menu";
  sidebarToggle.setAttribute("aria-label", label);
  sidebarToggle.title = label;
}
sidebarToggle.addEventListener("click", () => setSidebar(sidebar.classList.contains("closed")));
setSidebar(window.innerWidth > 700);
const view = gui.addFolder("View");
view
  .add(settings, "exaggeration", 1, 100, 1)
  .name("Vertical exaggeration")
  .onChange((v: number) => {
    // Same view of the same place: the rotation centre, on the terrain, moves
    // with the heights, and the camera keeps its direction and distance to it
    const ratio = v / shared.exaggeration.value;
    const offset = camera.position.clone().sub(controls.target);
    controls.target.z *= ratio;
    camera.position.copy(controls.target).add(offset);
    controls.update();
    viewToUrl();
    shared.exaggeration.value = v;
    for (const layer of lineLayers) layer.setExaggeration(v);
    iceDivides.setExaggeration(v);
    profileMarker.scale.z = v / 1000;
    iceCores.setExaggeration(v);
    aircraft.setExaggeration(v);
    eggs.setExaggeration(v);
    pointerMoved = true;
    requestRender();
  });
view.add(settings, "resetView").name("Reset view");
view.add(settings, "topView").name("Top view (map)");
view.add(settings, "edgeMove").name("Move at the edges of the view").onChange((on: boolean) => {
  localStorage.setItem(EDGE_KEY, on ? "on" : "off");
  if (!on) keyNavigation.setEdge(0, 0);
});
view.add(settings, "autoZoom").name("Fly to a clicked line").onChange((on: boolean) => localStorage.setItem(AUTO_ZOOM_KEY, on ? "on" : "off"));
view.add({ tutorial: () => tutorial.start() }, "tutorial").name("Tutorial (tour of the viewer)");
const layers = gui.addFolder("Layers");
layers
  .add(settings, "bed")
  .name("Bed (Bedmap3)")
  .onChange((v: boolean) => {
    bed.visible = v;
    bedRange.show(v);
    requestRender();
  });
// Options of a layer are only shown with it
const bedRange = layers
  .add(settings, "bedRange", 2000, 12000, 500)
  .name("Bed colour range (± m)")
  .onChange(updateBedRange)
  .show(settings.bed);
layers
  .add(settings, "surface")
  .name("Ice surface")
  .onChange((v: boolean) => {
    surface.visible = v;
    surfaceOpacity.show(v);
    requestRender();
  });
const surfaceOpacity = layers
  .add(settings, "surfaceOpacity", 0.05, 1, 0.05)
  .name("Surface opacity")
  .onChange((v: number) => {
    surface.material.uniforms.opacity.value = v;
    requestRender();
  })
  .show(settings.surface);
layers.add(settings, "velocity").name("Ice velocity (MEaSUREs)").onChange(showVelocity);
// Options of the velocity layer, right under it, only shown with it
const velocityOptions = layers.addFolder("Ice velocity options");
velocityOptions
  .add(settings, "velocityOpacity", 0.05, 1, 0.05)
  .name("Opacity")
  .onChange((v: number) => {
    if (velocity) velocity.material.uniforms.opacity.value = v;
    requestRender();
  });
velocityOptions.add(settings, "flowLines").name("Flow lines").onChange(showVelocity);
velocityOptions
  .add(settings, "animateFlow")
  .name("Animate flow")
  .onChange(() => {
    lastFlowUpdate = performance.now();
    requestRender();
  });
velocityOptions.hide();
layers
  .add(settings, "isochrones")
  .name("Isochrones")
  .onChange((v: boolean) => {
    isochrones.group.visible = v;
    showIsochroneLegend();
    pointerMoved = true;
    requestRender();
  });
layers
  .add(settings, "tracks")
  .name("Surface tracks")
  .onChange(showTracks);
layers
  .add(settings, "aircraft")
  .name("Survey aircraft")
  .onChange((v: boolean) => {
    aircraft.group.visible = v;
    // The tracks are their flight paths
    if (v) irhIndex.then((index) => tracks.load(IRH_URL, index)).catch(showError);
    lastAircraftUpdate = performance.now();
    requestRender();
  });
layers
  .add(settings, "iceCores")
  .name("Ice cores")
  .onChange((v: boolean) => {
    iceCores.group.visible = v;
    pointerMoved = true;
    requestRender();
  });
layers.add(settings, "groundingLines").name("Grounding lines (RAISED)").onChange(showGroundingLines);
// Options of the grounding lines, right under them, only shown with them
const groundingLineOptions = layers.addFolder("Grounding line options");
groundingLineOptions
  .add(settings, "groundingLineScenario", ["A", "B", "A and B"])
  .name("RAISED scenario")
  .onChange(showGroundingLines);
// The present grounding line, then the RAISED times from the youngest
for (const [age, , label] of GROUNDING_LINE_COLORS) {
  groundingLineOptions
    .add(settings.groundingLineTimes, String(age))
    .name(age === 0 ? "Present grounding line" : `Grounding line at ${label}`)
    .onChange(showGroundingLines);
}
groundingLineOptions.hide();
layers.add(settings, "iceDivides").name("Ice divides (IMBIE2)").onChange(showIceDivides);

function showGroundingLines(): void {
  const on = settings.groundingLines;
  groundingLines.setVisible(on);
  groundingLineOptions.show(on);
  document.getElementById("raised-attribution")!.hidden = !on;
  const apply = () => {
    groundingLines.scenario = settings.groundingLineScenario;
    groundingLines.hiddenAges.clear();
    for (const [age] of GROUNDING_LINE_COLORS) if (!settings.groundingLineTimes[age]) groundingLines.hiddenAges.add(age);
    groundingLines.applyFilter();
    for (const layer of groundingLines.lineLayers) layer.invalidate();
    showGroundingLineLegend();
    pointerMoved = true;
    requestRender();
  };
  if (on) terrainHeights.then((heights) => groundingLines.load(OUTLINES_URL, heights)).then(apply).catch(showError);
  apply();
}

function showGroundingLineLegend(): void {
  if (!settings.groundingLines) return groundingLineLegend.hide();
  const times = GROUNDING_LINE_COLORS.filter(([age]) => settings.groundingLineTimes[age]).map(
    ([, color, label]): [string, string, null] => [label, color, null],
  );
  const styles = CONFIDENCE_STYLES.map(([confidence, dash]): [string, string, [number, number] | null] => [confidence, "#8d97a3", dash]);
  groundingLineLegend.showLines("Grounding lines", [times, styles]);
}

function showIceDivides(): void {
  const on = settings.iceDivides;
  iceDivides.setVisible(on);
  document.getElementById("imbie-attribution")!.hidden = !on;
  if (on) {
    terrainHeights
      .then((heights) => iceDivides.load(OUTLINES_URL, heights))
      .then(() => {
        pointerMoved = true;
        requestRender();
      })
      .catch(showError);
  }
  pointerMoved = true;
  requestRender();
}

function showTracks(): void {
  tracks.group.visible = settings.tracks;
  if (settings.tracks) {
    irhIndex
      .then((index) => tracks.load(IRH_URL, index))
      .then(() => {
        tracks.lines.setLineWidth(Math.max(1, settings.lineWidth - 0.5));
        tracks.applyFilter(isochrones.hiddenDatasets);
        pointerMoved = true;
        requestRender();
      })
      .catch(showError);
  }
  pointerMoved = true;
  requestRender();
}
const isochroneFolder = gui.addFolder("Isochrones");
isochroneFolder
  .add(settings, "colorBy", COLOR_BY)
  .name("Colour by")
  .onChange((v: ColorBy) => {
    isochrones
      .setColorBy(v)
      .then(() => {
        showIsochroneLegend();
        requestRender();
      })
      .catch(showError);
  });
isochroneFolder
  .add(settings, "lineWidth", 1, 8, 0.5)
  .name("Line width")
  .onChange((v: number) => {
    isochrones.lines.setLineWidth(v);
    tracks.lines.setLineWidth(Math.max(1, v - 0.5));
    requestRender();
  });

// Filters: datasets and ages shown, filled once the index is loaded
const filters = gui.addFolder("Filter");
let filterQueued = false;
/** Apply the filters once, after all the checkboxes changed in this frame. */
function queueFilter(): void {
  if (filterQueued) return;
  filterQueued = true;
  requestAnimationFrame(() => {
    filterQueued = false;
    isochrones.applyFilter();
    tracks.applyFilter(isochrones.hiddenDatasets);
    for (const layer of lineLayers) layer.invalidate();
    showIsochroneLegend();
    clearHover();
    pointerMoved = true;
    requestRender();
  });
}

/** A sub-folder of checkboxes, all checked, with buttons to check or clear them all. */
function checkboxFolder<T>(title: string, items: [label: string, value: T][], hidden: Set<T>): void {
  const folder = filters.addFolder(title).close();
  const state: Record<string, boolean> = {};
  const all = (on: boolean) => () => {
    for (const c of boxes) c.setValue(on);
  };
  folder.add({ all: all(true) }, "all").name("Select all");
  folder.add({ none: all(false) }, "none").name("Clear all");
  const boxes = items.map(([label, value]) => {
    state[label] = !hidden.has(value);
    return folder
      .add(state, label)
      .onChange((on: boolean) => {
        if (on) hidden.delete(value);
        else hidden.add(value);
        queueFilter();
      });
  });
}

function buildFilters(index: IrhIndex): void {
  // Datasets by their citation, e.g. "Cavitte et al. (2020)"
  const labels = new Map<string, string>();
  for (const f of index.features) if (!labels.has(f.dataset)) labels.set(f.dataset, f.citation ?? f.dataset);
  const datasets = [...labels].sort((a, b) => a[1].localeCompare(b[1])).map(([name, label]): [string, string] => [label, name]);
  checkboxFolder("Datasets", datasets, isochrones.hiddenDatasets);
  // The ages, then the isochrones without age, then the basal unit (not an age, here for now)
  const rank = (k: AgeKey) => (typeof k === "number" ? 0 : k === null ? 1 : 2);
  const ages = [...new Set(index.features.map(ageKey))].sort((a, b) => rank(a) - rank(b) || (a as number) - (b as number));
  checkboxFolder(
    "Ages",
    ages.map((a): [string, AgeKey] => [a === "basal unit" ? "Basal unit" : a === null ? "No age" : `${a} ka`, a]),
    isochrones.hiddenAges,
  );
}

const light = gui.addFolder("Light");
light.add(settings, "lightAzimuth", 0, 360, 1).name("Azimuth").onChange(updateLight);
light.add(settings, "lightElevation", 5, 90, 1).name("Elevation").onChange(updateLight);
light.close();
const performanceFolder = gui.addFolder("Performance");
performanceFolder
  .add(settings, "quality", Object.keys(QUALITY))
  .name("Terrain mesh")
  .onChange((v: string) => {
    if (textureGrid && textureGrid.dx < 4) buildMeshes(QUALITY[v]);
  });
performanceFolder.add(settings, "lowResWhileMoving").name("Lower resolution while moving");
performanceFolder.close();

// Tour of the viewer, started by itself on the first visit (not in a private preview)
const openInMenu = (folder: GUI) => () => {
  setSidebar(true);
  folder.open();
  folder.domElement.scrollIntoView({ block: "nearest" });
};
const tutorial = new Tutorial([
  {
    title: "Welcome to the AntADatabase 3D viewer",
    text:
      "The lines under the ice surface are **radar isochrones**: layers inside the ice sheet that were " +
      "once the surface, all of the same age, traced by radar surveys. Below them, the bed of Antarctica " +
      "(Bedmap3). A short tour of the viewer?",
    prepare: () => resetView(),
  },
  {
    title: "Move around",
    text:
      "**Drag** to rotate and tilt, **Ctrl + drag** (or right-drag) to move over the ground, and " +
      "**scroll** to zoom towards the mouse. Or point at an **edge** of the view to move towards it. On a " +
      "touch screen: one finger rotates, two fingers zoom and move. The keys W A S D (or the arrows), Q and E, " +
      "and Z and X do the same, and 0 brings back the whole continent; this pad too.",
    target: () => document.getElementById("move-pad"),
  },
  {
    title: "Layers",
    text: "Show or hide the ice surface, the bed, the ice velocity, the grounding lines since the last ice age, the ice divides, the deep ice cores and the survey aircraft.",
    tip: "Decrease the surface opacity to discover the isochrones under the ice!",
    target: () => layers.domElement,
    prepare: () => {
      resetView();
      openInMenu(layers)();
    },
  },
  {
    title: "Point and click",
    text:
      "**Hover** a line for its details: its dataset, age and depth. **Click** an isochrone to fly to its " +
      "radar line and open its stratigraphy: all its isochrones, along the line, over the bed. Closing it " +
      "brings you back to where you were.",
  },
  {
    title: "Colours",
    text: "The isochrones are coloured by their age, in thousands of years (ka). The legend is here; the menu can also colour them by depth, fractional depth or elevation.",
    target: () => document.querySelector("#legends .legend:not([hidden])"),
  },
  {
    title: "Filter",
    text: "Choose the datasets and the ages shown. Every dataset can be downloaded from the Download page, with how to cite it.",
    target: () => filters.domElement,
    prepare: () => {
      resetView();
      openInMenu(filters)();
    },
  },
  {
    title: "Find the eggs!",
    text: "A few strange things are hidden somewhere in Antarctica, on the bed and in the ocean. Zoom in close: they are tiny from afar.",
    prepare: () => resetView(),
  },
]);

// Start: quick preview, then the detailed terrain
const loading = document.getElementById("loading")!;
if (!viewFromUrl()) resetView(true);
loadMeta(DATA_URL)
  .then(async (meta) => {
    document.getElementById("bedmap-doi")?.setAttribute("href", meta.doi);
    await loadLevel(meta, PREVIEW);
    if (!SUBMISSION && !tutorialSeen() && !document.body.classList.contains("clean")) tutorial.start();
    loading.textContent = "Loading detailed terrain…";
    await loadLevel(meta, DETAIL);
    loading.remove();
  })
  .catch((error) => {
    loading.textContent = String(error.message ?? error);
    console.error(error);
  });

showTracks();
irhIndex.then(buildFilters).catch(showError);
irhIndex
  .then((index) => isochrones.load(IRH_URL, index))
  .then(() => {
    isochrones.applyFilter();
    showIsochroneLegend();
    requestRender();
  })
  .catch(showError);

// Debugging from the browser console in development: window.viewer
if (import.meta.env.DEV) Object.assign(window, { viewer: { scene, camera, controls, keyNavigation, isochrones, tracks, iceCores, aircraft, renderer, bed, surface, settings, buildMeshes, groundingLines, iceDivides, profile, openProfileAt, tutorial } });
