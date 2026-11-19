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
import { ageKey, BASAL_UNIT_COLOR, COLOR_BY, isBasalUnit, Isochrones, lineKey, loadIndex, loadRegions, TrackBed, Tracks, type AgeKey, type ColorBy, type IrhIndex, type Feature, type Region, type Track } from "./irh";
import { closer, type Hit, type LineLayer } from "./lines";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { PivotField } from "./pivot";
import { BED_STOPS, SPEED, type Tick } from "./colormaps";
import { Legend } from "./legend";
import { KeyNavigation } from "./keys";
import { EggCounter } from "./eggcounter";
import { EasterEggs, ENDURANCE_AT } from "./eggs";
import { Sea } from "./sea";
import { onThemeChange, themeButton } from "./theme";
import { IceCores, polarStereographic, type IceCore } from "./icecores";
import { InTheWay } from "./inway";
import { Aircraft, flightLines, type FlightLine } from "./aircraft";
import { FlowField, FlowLines, loadSpeedTexture, loadVelocityMeta, LOG_SPEED_RANGE } from "./velocity";
import { SUBMISSION, startPreview } from "./preview";
import { Tutorial, askForTour, tutorialSeen } from "./tutorial";
import { CONFIDENCE_STYLES, GROUNDING_LINE_COLORS, GroundingLines, IceDivides, type HeightAt, type Scenario } from "./outlines";
import { ProfilePanel, type ProfilePoint, type Tooltip } from "./profile";
import { BEDMAP_DATASETS, BedmapLines, type BedmapTrack } from "./bedmap";
import { rangeSource } from "./sitedata";
import { normalise, regionSearch } from "./regionsearch";

// Data exported by anta_database.web: served with the app in development
// (web/public/data), or from the bucket when VITE_DATA_URL is set at build time
const DATA_ROOT = (import.meta.env.VITE_DATA_URL || `${import.meta.env.BASE_URL}data`).replace(/\/$/, "");
const DATA_URL = `${DATA_ROOT}/terrain`;
// The isochrones of the database, or of a submission in its private preview (?preview=<id>&key=<key>).
// The development viewer reads those of the next version of the data (VITE_IRH_DATA_URL)
const IRH_ROOT = (import.meta.env.VITE_IRH_DATA_URL || DATA_ROOT).replace(/\/$/, "");
const IRH_URL = SUBMISSION ? SUBMISSION.url : `${IRH_ROOT}/irh`;
// The page for teaching (lesson.ts): the viewer, with its settings not remembered
const LESSON = document.body.dataset.page === "lesson";
const VELOCITY_URL = `${DATA_ROOT}/velocity`;
const OUTLINES_URL = `${DATA_ROOT}/outlines`;
const BEDMAP_URL = `${DATA_ROOT}/bedmap`;
const BEDMAP_NO_PROFILE =
  "Profile not displayed for this survey: it has no line IDs, so the whole survey is stored as one line, many profiles one after the other.";
// Resolution of the quick preview, then of the shading texture
const PREVIEW = "8km";
const DETAIL = "2km";
// Mesh resolution of each quality setting; the shading is always from the texture
const QUALITY: Record<string, string> = { Fast: "8km", Detailed: "4km" };

// Renderer, scene, camera
const container = document.getElementById("app")!;
const renderer = new THREE.WebGLRenderer({ antialias: true });
// Full resolution when still, lower while the view moves. At most MAX_PIXELS drawn: an integrated
// GPU redrawing more, 30 times a second, can stall the whole desktop of a laptop
const MAX_PIXELS = 4e6; // a 2560×1440 window at ratio 1
let FULL_PIXEL_RATIO = 1;
let MOVING_PIXEL_RATIO = 1;
function setPixelRatios(): void {
  const fit = Math.sqrt(MAX_PIXELS / Math.max(1, window.innerWidth * window.innerHeight));
  FULL_PIXEL_RATIO = Math.max(0.5, Math.min(window.devicePixelRatio, 2, fit));
  MOVING_PIXEL_RATIO = Math.max(0.5, Math.min(0.75, FULL_PIXEL_RATIO), FULL_PIXEL_RATIO / 2);
}
setPixelRatios();
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
// Zooming to the cursor keeps the rotation centre on the line of sight (OrbitControls puts
// it there with screen space panning; otherwise, looking near the horizon, it leaves the
// centre behind and turns the camera towards it). Panning still moves along the ground:
// the up movement of the screen is replaced by the one over the ground
controls.screenSpacePanning = true;
const internals = controls as unknown as {
  _panOffset: THREE.Vector3;
  _sphericalDelta: THREE.Spherical;
  _dollyStart: THREE.Vector2;
  _panUp(distance: number, matrix: THREE.Matrix4): void;
  _handleMouseDownDolly(event: MouseEvent): void;
  _updateZoomParameters(x: number, y: number): void;
};
internals._panUp = (distance, matrix) => {
  const across = new THREE.Vector3().setFromMatrixColumn(matrix, 0);
  internals._panOffset.add(new THREE.Vector3().crossVectors(camera.up, across).multiplyScalar(distance));
};
// Zooming with the middle button: towards the pointer (OrbitControls gives it x for y)
internals._handleMouseDownDolly = (event) => {
  internals._updateZoomParameters(event.clientX, event.clientY);
  internals._dollyStart.set(event.clientX, event.clientY);
};
// Closest zoom (km), apart from the views of the eggs
const MIN_DISTANCE = 20;
controls.minDistance = MIN_DISTANCE;
controls.maxDistance = 15000;
controls.maxPolarAngle = 0.85 * Math.PI; // below the horizon too, to look under the sea

// Dragging rotates; with Ctrl (or Shift, Cmd) held, or the right button, it moves over the
// ground (OrbitControls). Pointing at an edge of the view also moves towards it (below)

// Rotate, zoom and pan around the middle of the ice under the centre of the
// screen, updated when an interaction starts. The centre moves along the line
// of sight, so the view doesn't jump.
let pivotField: PivotField | null = null;
// Heights of the terrain as drawn, once the first (coarse) level is loaded
let terrainReady: (heights: HeightAt) => void;
const terrainHeights = new Promise<HeightAt>((resolve) => (terrainReady = resolve));
// Zoomed out beyond ZOOMED_OUT (km from the centre), the view turns around the middle of the
// continent, the centre of the home view, wherever it is on the screen: the camera stays where
// it is when an interaction starts (rotations in the update of the controls, below). The zoom
// still goes towards the pointer
const ZOOMED_OUT = 4000;
let rotationAnchor: THREE.Vector3 | null = null;
function homeTarget(): THREE.Vector3 {
  const { x, y } = HOME_CENTRE;
  return new THREE.Vector3(x, y, pivotField ? (pivotField.heightAt(x, y) * shared.exaggeration.value) / 1000 : 0);
}
controls.addEventListener("start", () => {
  const far = camera.position.distanceTo(controls.target) >= ZOOMED_OUT;
  rotationAnchor = far ? homeTarget() : null;
  if (!pivotField) return;
  const direction = camera.getWorldDirection(new THREE.Vector3());
  const hit = pivotField.intersect(camera.position, direction, shared.exaggeration.value, 2 * controls.maxDistance);
  if (hit && hit.distanceTo(camera.position) >= controls.minDistance) {
    controls.target.copy(hit);
  } else {
    // Closer than the closest zoom, nothing ahead (the sky), or the camera in the ice, under
    // the sea or under the middle of the ice: the centre stays on the line of sight
    const distance = Math.max(camera.position.distanceTo(controls.target), controls.minDistance);
    controls.target.copy(camera.position).addScaledVector(direction, distance);
  }
});

// The camera can go through the ice and under the sea, but not deep into the rock: a step of
// the controls that would take it further into the bed than ROCK_TOLERANCE is undone, which
// stops the zoom there. Only steps going deeper are undone, so it can always come back out
// (in a narrow valley, or over mountains finer than the coarse grid of the heights)
const ROCK_TOLERANCE = 200; // m
/** How deep (m) a point (scene units) is under the bed, negative above it; null without data. */
function depthInRock(position: THREE.Vector3): number | null {
  const bed = pivotField?.smoothAt("bed", position.x, position.y) ?? null;
  return bed === null ? null : bed - (position.z * 1000) / shared.exaggeration.value;
}
const updateControls = controls.update.bind(controls);
const beforeUpdate = { position: new THREE.Vector3(), target: new THREE.Vector3(), quaternion: new THREE.Quaternion() };
controls.update = (deltaTime?: number | null) => {
  beforeUpdate.position.copy(camera.position);
  beforeUpdate.target.copy(controls.target);
  beforeUpdate.quaternion.copy(camera.quaternion);
  // Around the anchor: the rotation of this step taken from the controls, and applied to the
  // camera and its target together, around the vertical and the horizontal across the view
  const turn = internals._sphericalDelta;
  const [theta, phi] = rotationAnchor && !flight ? [turn.theta, turn.phi] : [0, 0];
  if (theta || phi) turn.set(turn.radius, 0, 0);
  let changed = updateControls(deltaTime);
  if (theta || phi) {
    // With damping, a part of the rotation left at each frame, as OrbitControls does
    const step = controls.enableDamping ? controls.dampingFactor : 1;
    // Angle of the view from the vertical (z is up, not y as for THREE.Spherical)
    const offset = camera.position.clone().sub(controls.target);
    const polar = Math.acos(THREE.MathUtils.clamp(offset.z / offset.length(), -1, 1));
    const tilt = THREE.MathUtils.clamp(polar + step * phi, 0.01, controls.maxPolarAngle) - polar;
    camera.updateMatrixWorld();
    const across = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0).setZ(0).normalize();
    const rotation = new THREE.Quaternion()
      .setFromAxisAngle(camera.up, step * theta)
      .multiply(new THREE.Quaternion().setFromAxisAngle(across, tilt));
    for (const point of [camera.position, controls.target]) point.sub(rotationAnchor!).applyQuaternion(rotation).add(rotationAnchor!);
    camera.lookAt(controls.target);
    // What is left of the rotation, damped
    const keep = controls.enableDamping ? 1 - controls.dampingFactor : 0;
    turn.theta = theta * keep;
    turn.phi = phi * keep;
    controls.dispatchEvent({ type: "change" });
    changed = true;
  }
  if (!changed || flight) return changed;
  const depth = depthInRock(camera.position);
  const depthBefore = depthInRock(beforeUpdate.position);
  if (depth !== null && depth > ROCK_TOLERANCE && (depthBefore === null || depth > depthBefore)) {
    camera.position.copy(beforeUpdate.position);
    controls.target.copy(beforeUpdate.target);
    camera.quaternion.copy(beforeUpdate.quaternion);
    internals._sphericalDelta.set(0, 0, 0);
    internals._panOffset.set(0, 0, 0);
    return false;
  }
  return changed;
};

// Crossing the ice surface: a layer of snow (white) for SNOW_DEPTH under it, then the ice
// (a light blue veil), drawn over the view
const SNOW_DEPTH = 8; // km in the scene, whatever the exaggeration, so it lasts a few zoom steps
const snowVeil = document.createElement("div");
snowVeil.id = "snow-veil";
container.appendChild(snowVeil);
let snowVeilShown = "";
function updateSnowVeil(): void {
  const p = camera.position;
  const surface = pivotField?.smoothAt("surface", p.x, p.y) ?? null;
  const rock = depthInRock(p);
  const depth = surface === null ? 0 : (surface * shared.exaggeration.value) / 1000 - p.z; // km
  let veil = "";
  if (settings.surface && depth > 0 && (rock === null || rock < 0)) {
    const snow = 0.92 * settings.surfaceOpacity * THREE.MathUtils.smoothstep(depth, 0, 0.15 * SNOW_DEPTH) * (1 - THREE.MathUtils.smoothstep(depth, 0.6 * SNOW_DEPTH, SNOW_DEPTH));
    const tint = 0.15 * settings.surfaceOpacity;
    veil = snow > tint ? `rgba(244, 248, 252, ${snow.toFixed(3)})` : `rgba(150, 200, 235, ${tint.toFixed(3)})`;
  }
  if (veil !== snowVeilShown) {
    snowVeil.style.background = veil;
    snowVeilShown = veil;
  }
}

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
// Opened with a view in the link (#view=...): kept, rather than the dataset of a preview or the
// home view. Read once at the start: the viewer writes its own view there after each move
const viewInUrl = /view=/.test(window.location.hash);

/**
 * Look at an area (EPSG:3031, m: xmin, xmax, ymin, ymax) from the direction of the home view: at
 * once, or flying there for ms.
 */
function frameBox([xmin, xmax, ymin, ymax]: number[], ms = 0): void {
  const x = (xmin + xmax) / 2000;
  const y = (ymin + ymax) / 2000;
  const size = Math.max(xmax - xmin, ymax - ymin) / 1000;
  const z = pivotField ? (pivotField.heightAt(x, y) * shared.exaggeration.value) / 1000 : 0;
  const target = new THREE.Vector3(x, y, z);
  const distance = THREE.MathUtils.clamp(4 * size, 200, controls.maxDistance);
  const position = target.clone().add(HOME_OFFSET.clone().setLength(distance));
  if (ms) {
    flyTo(target, position, ms);
    return;
  }
  controls.target.copy(target);
  camera.position.copy(position);
  controls.update();
  requestRender();
}

// Clicking a line flies to it, and a "Reset view" button pops up (gone once the view is reset,
// from it or the menu)
const backButton = document.createElement("button");
backButton.id = "back-view";
backButton.type = "button";
backButton.textContent = "↺ Reset view";
backButton.hidden = true;
backButton.addEventListener("click", () => resetView());
document.body.append(backButton);
function forgetZoom(): void {
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
  // Up away from the ground and down again on the way (km further than the straight path, at mid-way)
  hop: number;
  // Called on each frame with the progress of the flight (0 to 1, eased)
  onStep?: (k: number) => void;
} | null = null;

/**
 * Move the view smoothly to a rotation centre and camera position, so you see where you go;
 * onEnd once there (by default, the view goes in the address). With orbit, the camera turns
 * around the centre (its distance, tilt and direction change evenly) instead of going straight.
 * With animate, the animations go on meanwhile (far views, where they are cheap).
 */
function flyTo(target: THREE.Vector3, position: THREE.Vector3, ms = 800, onEnd = viewToUrl, orbit = false, animate = false, hop = 0): void {
  flight = { target0: controls.target.clone(), position0: camera.position.clone(), target, position, start: performance.now(), ms, onEnd, orbit, animate, hop };
  requestRender();
}

/**
 * Fly far, from close to close (to an egg found): out to an overview while leaving, then in
 * while arriving, the camera never closer to the ground than at both ends.
 */
function hopTo(target: THREE.Vector3, position: THREE.Vector3): void {
  const travel = Math.hypot(target.x - controls.target.x, target.y - controls.target.y);
  const ends = (camera.position.distanceTo(controls.target) + position.distanceTo(target)) / 2;
  // Seen from about as high as the way is long, at mid-way
  const hop = Math.max(0, Math.min(controls.maxDistance, 1.2 * travel) - ends);
  flyTo(target, position, THREE.MathUtils.clamp(1200 + 0.6 * travel, 1500, 4000), viewToUrl, false, true, hop);
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
  } else if (flight.hop > 0) {
    // The offset of the camera from the centre goes from one to the other, longer on the way
    const offset0 = flight.position0.clone().sub(flight.target0);
    const offset1 = flight.position.clone().sub(flight.target);
    const offset = offset0.clone().lerp(offset1, k);
    offset.setLength(offset0.length() + (offset1.length() - offset0.length()) * k + flight.hop * Math.sin(Math.PI * k));
    camera.position.copy(controls.target).add(offset);
  } else {
    camera.position.lerpVectors(flight.position0, flight.position, k);
  }
  flight.onStep?.(k);
  if (t === 1) {
    const { onEnd } = flight;
    flight = null;
    onEnd();
  }
}
controls.addEventListener("start", () => {
  // A flight changing the settings on the way (the surface opacity to an ice core) leaves them where it stopped
  if (flight?.onStep) saveSettings();
  flight = null;
});

// Closest the view of a clicked line comes (km from the camera to the middle of the line), so a
// short line isn't seen too close: higher zooms in less
const LINE_VIEW_MIN_DISTANCE = 1000;

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
  fadeSurfaceOnTheWay();
}

// Surface opacity at the end of a flight to a line or an ice core, so they show through
const SEEN_THROUGH_OPACITY = 0.7;

/**
 * On the flight just started, the ice surface (if shown) fades to at most SEEN_THROUGH_OPACITY in the
 * second half of the way, its slider following.
 */
function fadeSurfaceOnTheWay(): void {
  const from = settings.surfaceOpacity;
  if (!flight || !settings.surface || from <= SEEN_THROUGH_OPACITY) return;
  const end = flight.onEnd;
  flight.onStep = (k) => {
    settings.surfaceOpacity = THREE.MathUtils.lerp(from, SEEN_THROUGH_OPACITY, THREE.MathUtils.smoothstep(k, 0.4, 1));
    surface.material.uniforms.opacity.value = settings.surfaceOpacity;
    eggs.setIceSurface(settings.surface, settings.surfaceOpacity);
    surfaceOpacity.updateDisplay();
  };
  // Saved once there (or wherever it stopped, see the start of the controls)
  flight.onEnd = () => {
    surfaceOpacity.setValue(SEEN_THROUGH_OPACITY);
    end();
  };
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
  setPixelRatios();
  if (strained) FULL_PIXEL_RATIO = MOVING_PIXEL_RATIO;
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
  Object.assign(stats.dom.style, { left: "auto", right: "10px", top: "auto", bottom: "150px" });
  stats.dom.id = "stats";
  document.body.appendChild(stats.dom);
}

// Flow lines move at most 30 times per second, to spare the GPU
const FLOW_INTERVAL = 1000 / 30;
let lastFlowUpdate = 0;
// The waves of the sea: their own clock, stopped while they are not drawn
let lastSeaUpdate = 0;
let lastEggUpdate = 0;
let seaTime = 0;

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
  return !strained && document.hasFocus() && performance.now() - lastInput < IDLE_MS;
}

// Guard of the GPU: when the animated frames come slower than STRAINED_MS on median, STRAINED_CHECKS
// times in a row (the GPU can't keep up, and the desktop may soon freeze with it), what moves is
// hidden (the aircraft, the animated flow lines; the sea and the eggs stand still) and the
// resolution drops, with a message. They come back after a pause (RETRY_MS, while the page has the
// focus), and are watched again: hidden again soon after, the next pause is twice as long (up to
// MAX_RETRY_MS)
const STRAINED_MS = 150; // about 7 images per second
const STRAINED_CHECKS = 2; // of 30 frames each
let strainedChecks = 0;
const RETRY_MS = 30_000;
const MAX_RETRY_MS = 5 * 60_000;
const frameGaps: number[] = [];
let strained = false;
let strainedAt = 0;
let resumedAt = -Infinity;
let retryMs = RETRY_MS;
function watchFrame(gap: number, animated: boolean): void {
  if (strained || !animated || gap > 2000) return; // a gap of seconds: a tab in the background
  frameGaps.push(gap);
  if (frameGaps.length < 30) return;
  const median = [...frameGaps].sort((a, b) => a - b)[15];
  frameGaps.length = 0;
  strainedChecks = median < STRAINED_MS ? 0 : strainedChecks + 1;
  if (strainedChecks < STRAINED_CHECKS) return;
  strainedChecks = 0;
  const now = performance.now();
  retryMs = now - resumedAt < 2 * RETRY_MS ? Math.min(2 * retryMs, MAX_RETRY_MS) : RETRY_MS;
  strained = true;
  strainedAt = now;
  FULL_PIXEL_RATIO = MOVING_PIXEL_RATIO;
  showAnimated();
  showError(
    new Error(
      `The graphics card can't keep up (${Math.round(1000 / median)} images per second): the aircraft and the flow lines are hidden, and the resolution lowered. They come back in ${Math.round(retryMs / 1000)} s.`,
    ),
  );
}

/** After the pause of the guard: the animations and the full resolution back. */
function resumeAfterStrain(now: number): void {
  if (!strained || now - strainedAt < retryMs || !document.hasFocus()) return;
  strained = false;
  resumedAt = now;
  frameGaps.length = 0;
  setPixelRatios();
  showAnimated();
  requestRender();
}

/** The layers that move, shown as set in the menu, but hidden while the guard holds. */
function showAnimated(): void {
  aircraft.group.visible = settings.aircraft && !strained;
  if (flowLines) flowLines.mesh.visible = settings.velocity && settings.flowLines && !(strained && settings.animateFlow);
  lastAircraftUpdate = lastFlowUpdate = performance.now();
  requestRender();
}

let lastFrame = 0;
let lastRender = 0;
let animatedLastFrame = false;
const GLIDE_FRAME_MS = 1000 / 30;
// Whether the animations (aircraft, flow lines) go on while moving with the keys or the pad, and
// while moving at an edge of the view (false: paused, lighter for a laptop)
const ANIMATE_WITH_KEYS = false;
const ANIMATE_AT_EDGES = true;
renderer.setAnimationLoop((time: number) => {
  stepFlight(performance.now());
  updateEdgeMove(performance.now());
  keyNavigation.update(Math.min(0.1, (time - lastFrame) / 1000));
  watchFrame(time - lastFrame, animatedLastFrame);
  resumeAfterStrain(performance.now());
  lastFrame = time;
  controls.update();
  updateSnowVeil();
  // Moving by itself (a flight, keys, the pad or an edge): drawn at most 30 times a second, and
  // without the animations (but during the flights home and to the map, and as set above for
  // the keys and the edges), so a laptop keeps up however long it lasts
  const gliding = flight !== null || keyNavigation.active;
  const paused = (keyNavigation.keysActive && !ANIMATE_WITH_KEYS) || (keyNavigation.edgeActive && !ANIMATE_AT_EDGES);
  const animate = animating() && (flight ? flight.animate : !paused);
  animatedLastFrame = animate;
  stopMomentum();
  previousCamera.copy(camera.position);
  updateHover();
  if (!animate) {
    // Go on from where they stopped, without a jump
    lastAircraftUpdate = lastFlowUpdate = lastSeaUpdate = lastEggUpdate = time;
  }
  // The moving eggs, only when the camera is near one of them
  if (animate && eggs.near(camera) && time - lastEggUpdate >= FLOW_INTERVAL) {
    eggs.update(Math.min(0.1, (time - lastEggUpdate) / 1000));
    lastEggUpdate = time;
    needsRender = true;
  }
  if (animate && sea.mesh.visible && time - lastSeaUpdate >= FLOW_INTERVAL) {
    seaTime += Math.min(0.1, (time - lastSeaUpdate) / 1000);
    sea.setTime(seaTime);
    lastSeaUpdate = time;
    needsRender = true;
  }

  if (animate && aircraft.group.visible && time - lastAircraftUpdate >= FLOW_INTERVAL) {
    const dt = Math.min(0.1, (time - lastAircraftUpdate) / 1000);
    aircraft.update(dt, camera, renderer.domElement.clientHeight);
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
  // Dashes of the grounding lines: the same length on screen at any zoom
  if (groundingLines.group.visible) groundingLines.setDashPixel(pixelAtTarget());
  // The lines in the way of the open profile fade: the depth of its curtain first
  // The eggs far away aren't drawn
  if (eggs.group.visible) eggs.cull(camera);
  inTheWay.render(renderer, camera);
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
// The sea at 0 m (menu, Miscellaneous), under the ice shelves: drawn before them
const heightUniforms = (m: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>) => {
  const u = m.material.uniforms;
  return { heightTex: u.heightTex, texOrigin: u.texOrigin, texDx: u.texDx, texSize: u.texSize };
};
const sea = new Sea(heightUniforms(bed), heightUniforms(surface));
sea.mesh.renderOrder = 0.5;
scene.add(sea.mesh);

// Isochrones, and their radar tracks at the surface (loaded when first shown)
const isochrones = new Isochrones();
const tracks = new Tracks();
// After the Bedmap tracks (order 3), on top of them
tracks.group.renderOrder = 4;
// Bed picks of the isochrone datasets along their tracks; off at start
const trackBed = new TrackBed();
trackBed.group.visible = false;
const lineLayers: LineLayer[] = [isochrones.lines, tracks.lines, trackBed.lines];
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
// Bedmap survey lines: tracks at the surface and the measured bed along them; off at start
const bedmap = new BedmapLines();
for (const layer of [bedmap.tracks, bedmap.bed]) {
  layer.group.visible = false;
  layer.setResolution(window.innerWidth, window.innerHeight);
  layer.setExaggeration(shared.exaggeration.value);
  lineLayers.push(layer);
  scene.add(layer.group);
}

const irhIndex = SUBMISSION
  ? startPreview(document.getElementById("preview")!).then((status) => {
      // Often long after the start, the submission being compiled: flying there from the home view
      if (status.bbox && !viewInUrl) {
        submissionBox = status.bbox;
        frameBox(submissionBox, 1500);
      }
      import("./preview-download")
        .then(({ showDownload }) => showDownload(document.getElementById("preview-download")!, SUBMISSION!.url, status))
        .catch(console.error);
      return loadIndex(IRH_URL);
    })
  : loadIndex(IRH_URL);

// In a private preview, the profile window opens below its panels (progress, download) rather
// than over them, but keeps some height for the plot (the panels can be reduced). Enlarged, it
// covers them all as anywhere else
const PROFILE_TOP = 56; // px, as #profile in style.css
const PROFILE_MIN_HEIGHT = 180;
function placeProfile(): void {
  const panel = document.getElementById("profile");
  const stack = document.getElementById("preview-stack")!;
  if (!panel) return;
  const shown = [...stack.children].some((c) => !(c as HTMLElement).hidden);
  const below = shown ? stack.getBoundingClientRect().bottom + 10 : PROFILE_TOP;
  const top = Math.max(PROFILE_TOP, Math.min(below, window.innerHeight - PROFILE_MIN_HEIGHT - 12));
  panel.style.setProperty("--profile-top", `${Math.round(top)}px`);
}
if (SUBMISSION) {
  const stack = document.getElementById("preview-stack")!;
  new ResizeObserver(placeProfile).observe(stack);
  // The panels shown or hidden, and their content changing
  new MutationObserver(placeProfile).observe(stack, { subtree: true, childList: true, attributes: true, attributeFilter: ["hidden", "class"] });
  window.addEventListener("resize", placeProfile);
}

// Deep ice core sites
const iceCores = new IceCores();
iceCores.setExaggeration(shared.exaggeration.value);
iceCores.group.visible = false; // off by default
scene.add(iceCores.group);

// Survey aircraft flying along the tracks, with their radar pulses
// One per dataset, flying the legs of its tracks, found once the tracks are loaded
const AIRCRAFT_DATASETS = ["Franke_2025", "Sanderson_2024", "Yan_2025", "Wang_2023", "Cavitte_2020", "Bodart_2021", "Bodart_2025b"];
let legs: Map<string, FlightLine[]> | null = null;
// The index is loaded, and the filters of the last visit restored: what to show is known
let filtersReady: Promise<void>;
function aircraftLegs(dataset: string): FlightLine[] {
  if (!tracks.lines.loaded) return [];
  legs ??= flightLines(
    tracks.lines.featureCount,
    (f) => tracks.lines.featureSegments(f),
    (f) => tracks.tracks[f]?.dataset ?? "",
  );
  return legs.get(dataset) ?? [];
}
/** The tracks of the datasets shown (with the filters), for themselves and for the aircraft. */
function loadTracks(): void {
  filtersReady
    .then(() => (settings.tracks || settings.aircraft ? tracks.load(isochrones.datasetsNotHidden()) : undefined))
    .then(() => {
      legs = null;
      tracks.lines.setLineWidth(Math.max(1, settings.lineWidth - 1));
      tracks.applyFilter(isochrones);
      pointerMoved = true;
      requestRender();
    })
    .catch(showError);
}
const aircraft = new Aircraft(
  AIRCRAFT_DATASETS,
  aircraftLegs,
  (f) => tracks.lines.isHidden(f),
  (x, y) => pivotField?.bedAt(x, y) ?? null,
);
aircraft.setExaggeration(shared.exaggeration.value);
scene.add(aircraft.group);

// Hidden things to find
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
  (x, y) => pivotField?.smoothAt("surface", x, y) ?? null,
  { heightTex: u.heightTex, texOrigin: u.texOrigin, texDx: u.texDx, texSize: u.texSize, exaggeration: shared.exaggeration, ...bedMesh },
);
eggs.setExaggeration(shared.exaggeration.value);
eggs.onLoaded = requestRender;
// Not in a private preview, for the submitted dataset only
if (!SUBMISSION) scene.add(eggs.group);
/** Fly to the view of an egg (eggs.ts): out and in again if it is far. */
function flyToEgg(id: string): void {
  const where = eggs.whereIs(id);
  if (!where) return;
  // Closer than the usual closest zoom for the small ones (the glass of Lorius), until zoomed out
  controls.minDistance = Math.min(MIN_DISTANCE, 0.7 * where.position.distanceTo(where.target));
  hopTo(where.target, where.position);
  backButton.hidden = false;
}
controls.addEventListener("end", () => {
  if (camera.position.distanceTo(controls.target) > MIN_DISTANCE) controls.minDistance = MIN_DISTANCE;
});
/**
 * A view of a whole ice core, from the bottom of the borehole to the surface: from the side
 * the camera is on, a bit above the horizontal so the view stays above the ice surface
 * (seen through it).
 */
function flyToCore(core: IceCore): void {
  const [x, y] = polarStereographic(core.lat, core.lon);
  const scale = shared.exaggeration.value / 1000;
  const target = new THREE.Vector3(x, y, (core.elevation - core.depth / 2) * scale);
  const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const distance = THREE.MathUtils.clamp((0.75 * core.depth * scale) / tan, 2 * MIN_DISTANCE, controls.maxDistance);
  const tilt = THREE.MathUtils.degToRad(25);
  let heading = Math.atan2(camera.position.y - y, camera.position.x - x);
  if (Math.hypot(camera.position.x - x, camera.position.y - y) < 1) heading = -Math.PI / 2;
  const offset = new THREE.Vector3(Math.cos(heading) * Math.cos(tilt), Math.sin(heading) * Math.cos(tilt), Math.sin(tilt));
  hopTo(target, target.clone().add(offset.multiplyScalar(distance)));
  backButton.hidden = false;
  fadeSurfaceOnTheWay();
}

const eggCounter = SUBMISSION ? null : new EggCounter(eggs.names, flyToEgg);

/** The eggs and their counter, shown or not (menu, Miscellaneous). */
const ENDURANCE_CLEARING = 45; // km

function showEggs(): void {
  eggs.group.visible = settings.eggs;
  // Clearer water over the wreck of the Endurance, to see it through the waves
  sea.setClearing(ENDURANCE_AT[0], ENDURANCE_AT[1], settings.eggs ? ENDURANCE_CLEARING : 0);
  if (eggCounter) eggCounter.el.hidden = !settings.eggs;
  pointerMoved = true;
  requestRender();
}

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
    material.uniforms.speedScale.value = speedScale();
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
  })();
  return velocityLoading;
}

/** Speed (m/a) of the scale of the ice speed colours, from the contrast in the menu. */
function speedScale(): number {
  return 10 ** -settings.velocityContrast;
}

/** Ticks of the ice speed legend, along log(1 + speed / speedScale). */
function speedTicks(): Tick[] {
  const s = speedScale();
  const max = 10 ** LOG_SPEED_RANGE.y;
  const t = (v: number) => Math.log1p(v / s) / Math.log1p(max / s);
  const out: Tick[] = [
    { t: 0, label: "0" },
    { t: 1, label: max.toLocaleString("en") },
  ];
  for (const v of [0.01, 0.1, 1, 10, 100, 1000]) {
    const tick = { t: t(v), label: v.toLocaleString("en") };
    if (tick.t > 0.06 && out.every((o) => Math.abs(o.t - tick.t) > 0.09)) out.push(tick);
  }
  return out.sort((a, b) => a.t - b.t);
}

/** Flow lines live around the centre of the view, over about the visible width. */
function updateFlowArea(): void {
  if (!flowLines) return;
  const halfHeight = camera.position.distanceTo(controls.target) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  flowLines.setArea(new THREE.Vector2(controls.target.x, controls.target.y), 1.2 * halfHeight * Math.max(1, camera.aspect));
}

function showVelocity(): void {
  const on = settings.velocity;
  if (on) loadDetail("surface").catch(showError);
  if (on && !velocity) loadVelocity().then(showVelocity).catch(showError);
  if (velocity) velocity.visible = on;
  showAnimated();
  velocityOptions.show(on);
  if (on && velocity) {
    velocityLegend.show("Ice speed (m/a)", SPEED, speedTicks());
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
  if (event.pointerType === "touch") {
    // The tooltip of a long press stays while the finger rests; moving it away cancels the press
    if (touchHold && Math.hypot(event.clientX - touchHold.x, event.clientY - touchHold.y) > LONG_PRESS_PX) cancelLongPress();
    if (touchTip) return;
  }
  pointer = event.buttons ? null : { x: event.offsetX, y: event.offsetY };
  pointerMoved = true;
});
renderer.domElement.addEventListener("pointerleave", (event) => {
  if (event.pointerType === "touch") return; // after each touch: the tooltip of a long press stays
  pointer = null;
  pointerMoved = true;
});

// No hover on a touch screen: a long press on a line (or anything with a tooltip) shows its
// tooltip, as a hover would, until the next touch
const LONG_PRESS_MS = 450;
const LONG_PRESS_PX = 8;
let touchHold: { x: number; y: number; timer: number } | null = null;
let touchTip = false;
const touches = new Set<number>();
function cancelLongPress(): void {
  if (touchHold) clearTimeout(touchHold.timer);
  touchHold = null;
}
renderer.domElement.addEventListener("pointerdown", (event) => {
  if (event.pointerType !== "touch") return;
  touches.add(event.pointerId);
  cancelLongPress();
  if (touchTip) {
    touchTip = false;
    pointer = null;
    pointerMoved = true;
  }
  if (touches.size > 1) return; // pinching
  const [x, y] = [event.offsetX, event.offsetY];
  touchHold = {
    x: event.clientX,
    y: event.clientY,
    timer: window.setTimeout(() => {
      touchHold = null;
      touchTip = true;
      pressed = null; // not a click
      pointer = { x, y };
      pointerMoved = true;
      lastViewChange = 0; // shown at once, the view hardly moved
      requestRender();
    }, LONG_PRESS_MS),
  };
});
for (const type of ["pointerup", "pointercancel"] as const) {
  renderer.domElement.addEventListener(type, (event) => {
    if (event.pointerType !== "touch") return;
    touches.delete(event.pointerId);
    cancelLongPress();
  });
}
renderer.domElement.addEventListener("contextmenu", (event) => event.preventDefault());

function formatAge(f: Feature): string {
  if (f.age_ka === null) return "not assigned";
  if (f.age_min_ka !== undefined && f.age_max_ka !== undefined) return `${f.age_min_ka}–${f.age_max_ka} ka (${rangeSource(f)})`;
  return f.age_unc_ka === null ? `${f.age_ka} ka` : `${f.age_ka} ± ${f.age_unc_ka} ka`;
}

/** Where the elevation of an isochrone comes from, when not only from SURF_ELEV. */
function surfaceNote(surface: string): string {
  return surface === "Bedmap3"
    ? "Elevation from depth and the Bedmap3 surface"
    : "Elevation from depth and SURF_ELEV, with Bedmap3 grid filling";
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
  if (f.surface !== "SURF_ELEV") notes.push(surfaceNote(f.surface));
  if (fdepth !== null && f.thickness === "Bedmap3") notes.push("Fractional depth with the Bedmap3 ice thickness");
  showTooltip(f.citation ?? f.dataset, rows, notes.length ? notes.join(". ") : null);
}

function trackBedTooltip(t: Track, hit: Hit): void {
  showTooltip(
    t.citation ?? t.dataset,
    [
      ["Line ID", t.line_id],
      ["Bed elevation", `${Math.round(trackBed.bedAt(hit))} m`],
    ],
    "Bed picked along the line (not the Bedmap3 grid), averaged along the line",
  );
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
  showTooltip(
    t.citation ?? t.dataset,
    rows,
    t.surface === "SURF_ELEV" ? null : t.surface === "Bedmap3" ? "Surface elevation from Bedmap3" : "Surface elevation from SURF_ELEV, with Bedmap3 grid filling",
  );
}

function iceCoreTooltip(core: IceCore): void {
  const rows: [string, string][] = [
    ["Position", `${Math.abs(core.lat).toFixed(2)}° S, ${Math.abs(core.lon).toFixed(2)}° ${core.lon < 0 ? "W" : "E"}`],
    ["Surface elevation", `${core.elevation} m`],
    ["Drilled depth", `${core.depth} m`],
  ];
  if (core.bottomAge) rows.push(["Oldest ice", core.bottomAge]);
  rows.push(["Drilled", core.drilled]);
  showTooltip(`${core.site} ice core`, rows, `${core.reference}. Click to zoom on it.`);
}

function showTooltip(heading: string, rows: [string, string][], footnote: string | null, picture?: string): void {
  tooltip.replaceChildren();
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = heading;
  tooltip.append(title);
  if (picture) {
    // Our own SVG (eggs.ts), not data
    const figure = document.createElement("div");
    figure.className = "picture";
    figure.innerHTML = picture;
    tooltip.append(figure);
  }
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
  const [width, height] = [window.innerWidth, window.innerHeight];
  const [gap, margin] = [16, 8];
  // Too tall for the window (a long story with a picture): wider, the picture beside the text,
  // then smaller text
  tooltip.classList.remove("wide", "compact");
  if (tooltip.offsetHeight > height - 2 * margin && tooltip.querySelector(".picture")) tooltip.classList.add("wide");
  if (tooltip.offsetHeight > height - 2 * margin) tooltip.classList.add("compact");
  const [w, h] = [tooltip.offsetWidth, tooltip.offsetHeight];
  // Right of the pointer if it fits, else left of it, else on the side with more room
  let left = x + gap + w <= width - margin ? x + gap : x - gap - w >= margin ? x - gap - w : width - x > x ? x + gap : x - gap - w;
  // Below, else above, else beside it, centred on its height: always within the window
  let top = y + gap + h <= height - margin ? y + gap : y - gap - h >= margin ? y - gap - h : y - h / 2;
  left = THREE.MathUtils.clamp(left, margin, Math.max(margin, width - w - margin));
  top = THREE.MathUtils.clamp(top, margin, Math.max(margin, height - h - margin));
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
  const egg = pointer && !core && !SUBMISSION && settings.eggs ? eggs.pick(camera, pointer.x, pointer.y, canvas.clientWidth, canvas.clientHeight) : null;
  // Closest line of all layers
  let best: { layer: LineLayer; hit: Hit } | null = null;
  if (pointer && !core && !egg) {
    for (const layer of lineLayers) {
      if ((layer === isochrones.lines || layer === bedmap.bed || layer === trackBed.lines) && isochronesUnderIce()) continue;
      const hit = layer.pick(camera, pointer.x, pointer.y, canvas.clientWidth, canvas.clientHeight);
      if (hit && (!best || closer(hit.distance, hit.depth, best.hit))) best = { layer, hit };
    }
  }
  if (core && pointer) {
    iceCoreTooltip(core);
    placeTooltip(pointer.x, pointer.y);
  } else if (egg && pointer) {
    showTooltip(egg.title, egg.rows, eggCounter?.has(egg.id) ? egg.note : `${egg.note} Click to collect it.`, egg.picture);
    placeTooltip(pointer.x, pointer.y);
  } else if (!best || !pointer) {
    tooltip.style.visibility = "hidden";
  } else {
    if (best.layer === tracks.lines) trackTooltip(tracks.tracks[best.hit.feature], best.hit);
    else if (best.layer === trackBed.lines) trackBedTooltip(tracks.tracks[best.hit.feature], best.hit);
    else if (best.layer === isochrones.lines) isochroneTooltip(isochrones.features[best.hit.feature], best.hit);
    else if (best.layer === iceDivides.lines) showTip(iceDivides.describe(best.hit));
    else if (best.layer === bedmap.tracks || best.layer === bedmap.bed) showTip(bedmap.describe(best.layer, best.hit));
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

// What hides the open profile: its curtain, to fade the lines in front of it
const inTheWay = new InTheWay();
inTheWay.setExaggeration(shared.exaggeration.value);
isochrones.lines.setInTheWay(inTheWay.uniforms);
tracks.lines.setInTheWay(inTheWay.uniforms);
trackBed.lines.setInTheWay(inTheWay.uniforms);
bedmap.tracks.setInTheWay(inTheWay.uniforms);
bedmap.bed.setInTheWay(inTheWay.uniforms);

const profile = new ProfilePanel({
  colorOf: (feature) => isochrones.ageColor(feature),
  describeLayer: (feature, depth, point) => {
    const f = isochrones.features[feature];
    const rows: [string, string][] = isBasalUnit(f) ? [["Layer", "Basal unit (no age)"]] : [["Age", formatAge(f)]];
    rows.push(["Elevation", `${Math.round(point.surface - depth)} m`], ["Depth", `${Math.round(depth)} m`]);
    if (point.thickness > 0) rows.push(["Fractional depth", (depth / point.thickness).toFixed(2)]);
    rows.push(["Line ID", f.line_id]);
    if (!isBasalUnit(f)) rows.push(["IRH", String(f.irh)]);
    return { title: f.citation ?? f.dataset, rows, note: f.surface !== "SURF_ELEV" ? surfaceNote(f.surface) : null };
  },
  showTooltip: (tip, x, y) => {
    showTip(tip);
    placeTooltip(x, y);
  },
  hideTooltip: () => (tooltip.style.visibility = "hidden"),
  onPoint: markProfilePoint,
  // Closing the stratigraphy only ends the selection of its line: the view stays
  onClose: () => {
    selectLine(null);
    inTheWay.setCurtain(null);
  },
  describeLine: (line, layers) => {
    if ("survey" in line) {
      // A Bedmap survey line
      const survey = bedmap.index?.surveys[(line as BedmapTrack).survey];
      const rows: [string, string][] = [
        ["Line ID", line.line_id],
        ["Survey", `${(line as BedmapTrack).survey} (${line.dataset})`],
      ];
      if (survey?.institute) rows.push(["Institute", survey.institute]);
      if (survey?.["acquisition year"]) rows.push(["Acquisition year", String(survey["acquisition year"])]);
      if (survey?.["radar instrument"]) rows.push(["Radar", survey["radar instrument"]]);
      if (survey?.["DOI dataset"]) rows.push(["Data", survey["DOI dataset"]]);
      return rows;
    }
    const rows: [string, string][] = [["Line ID", line.line_id]];
    const features = layers.map((f) => isochrones.features[f]);
    const dated = features.filter((f) => !isBasalUnit(f) && f.age_ka !== null).sort((a, b) => a.age_ka! - b.age_ka!);
    const ages = dated.map((f) => `${formatAge(f)} (IRH ${f.irh})`);
    const undated = features.filter((f) => !isBasalUnit(f) && f.age_ka === null).length;
    if (undated) ages.push(`${undated} without age`);
    rows.push([`Isochrones on this line (${features.filter((f) => !isBasalUnit(f)).length})`, ages.join(", ") || "none"]);
    if (features.some(isBasalUnit)) rows.push(["Also on this line", "a basal unit layer"]);
    return rows;
  },
});

/**
 * Draw out the track and the isochrones of the line of the open profile (wider, with a glow,
 * through the ice), the other isochrones and tracks thinner and fainter; all as before with null.
 */
function selectLine(line: Track | BedmapTrack | null): void {
  const same = (f: { dataset: string; line_id: string }) => line !== null && f.dataset === line.dataset && f.line_id === line.line_id;
  const indices = <T extends { dataset: string; line_id: string }>(items: T[]) => items.flatMap((f, i) => (same(f) ? [i] : []));
  isochrones.lines.setSelection(indices(isochrones.features), true);
  tracks.lines.setSelection(indices(tracks.tracks), true);
  // Bedmap lines by track: two surveys can have the same line ID
  const bedmapTrack = line && "survey" in line ? (bedmap.index?.tracks.indexOf(line) ?? -1) : -1;
  for (const layer of [bedmap.tracks, bedmap.bed]) layer.setSelection(bedmapTrack >= 0 ? [bedmapTrack] : [], true);
  requestRender();
}

// Lines (tracks) by dataset and line ID, with their profiles
const linesById = new Map<string, Track>();
irhIndex.then((index) => {
  for (const t of index.tracks) linesById.set(`${t.dataset}\u0000${t.line_id}`, t);
});

/** Open the profile of the isochrone, track or Bedmap line under a pixel of the canvas, if any. */
function openProfileAt(x: number, y: number): void {
  const canvas = renderer.domElement;
  let best: { layer: LineLayer; hit: Hit } | null = null;
  for (const layer of [isochrones.lines, tracks.lines, trackBed.lines, bedmap.tracks, bedmap.bed]) {
    if (!layer.group.visible || ((layer === isochrones.lines || layer === bedmap.bed || layer === trackBed.lines) && isochronesUnderIce())) continue;
    const hit = layer.pick(camera, x, y, canvas.clientWidth, canvas.clientHeight);
    if (hit && (!best || closer(hit.distance, hit.depth, best.hit))) best = { layer, hit };
  }
  if (!best) return;
  let track: Track | BedmapTrack | undefined;
  let baseUrl = IRH_URL;
  if (best.layer === bedmap.tracks || best.layer === bedmap.bed) {
    track = bedmap.index?.tracks[best.hit.feature];
    // Only lines with their own line ID have a profile: a survey stored as one line is many
    // profiles one after the other. Still selected, zoomed to, with its details
    if (track && !track.profile) {
      selectLine(track);
      inTheWay.setCurtain(null);
      profile.open(track, BEDMAP_URL, BEDMAP_NO_PROFILE).catch(showError);
      const segments = bedmap.tracks.featureSegments(best.hit.feature);
      if (settings.autoZoom && segments) {
        const n = segments.length / 3;
        const [x, y] = [new Float32Array(n), new Float32Array(n)];
        for (let i = 0; i < n; i++) [x[i], y[i]] = [segments[3 * i], segments[3 * i + 1]];
        zoomToLine(x, y);
      }
      return;
    }
    baseUrl = BEDMAP_URL;
  } else {
    const line = best.layer === tracks.lines || best.layer === trackBed.lines ? tracks.tracks[best.hit.feature] : isochrones.features[best.hit.feature];
    track = linesById.get(`${line.dataset}\u0000${line.line_id}`);
  }
  if (!track) return;
  const opened = track;
  selectLine(opened);
  inTheWay.setCurtain(null);
  profile
    .open(opened, baseUrl)
    .then(() => {
      if (profile.line === opened) inTheWay.setCurtain(profile.curtain());
      // Unless another line was clicked meanwhile
      if (settings.autoZoom && profile.line === opened) {
        const { x, y } = profile.points();
        zoomToLine(x, y);
      }
    })
    .catch(showError);
}

/** Collect the egg under a click, if not yet, and fly to a view of it; whether there was one. */
function collectEgg(event: PointerEvent): boolean {
  if (!eggCounter || !settings.eggs) return false;
  const canvas = renderer.domElement;
  if (iceCores.pick(camera, event.offsetX, event.offsetY, canvas.clientWidth, canvas.clientHeight)) return false;
  const egg = eggs.pick(camera, event.offsetX, event.offsetY, canvas.clientWidth, canvas.clientHeight);
  if (!egg) return false;
  // Collected the first time; each time, a good view of it
  if (!eggCounter.has(egg.id)) eggCounter.collect(egg.id, egg.title, event.clientX, event.clientY);
  flyToEgg(egg.id);
  pointerMoved = true;
  requestRender();
  return true;
}

// A click (not a drag) on a line opens its profile
let pressed: { x: number; y: number; time: number } | null = null;
renderer.domElement.addEventListener("pointerdown", (event) => {
  pressed = event.button === 0 ? { x: event.clientX, y: event.clientY, time: performance.now() } : null;
});
renderer.domElement.addEventListener("pointerup", (event) => {
  const click = pressed && Math.hypot(event.clientX - pressed.x, event.clientY - pressed.y) < 5 && performance.now() - pressed.time < 500;
  pressed = null;
  if (!click) return;
  const canvas = renderer.domElement;
  const core = iceCores.group.visible ? iceCores.pick(camera, event.offsetX, event.offsetY, canvas.clientWidth, canvas.clientHeight) : null;
  if (core) flyToCore(core);
  else if (!collectEgg(event)) openProfileAt(event.offsetX, event.offsetY);
});

let maskTexture: THREE.DataTexture | null = null;
// Grid and data coverage of the current height textures, to rebuild the meshes
const textureGrid: Partial<Record<TerrainLayer, Grid>> = {};
const coverage: Partial<Record<TerrainLayer, Uint8Array>> = {};
type TerrainLayer = "bed" | "surface";
// The detailed height textures, each fetched when its layer is first shown
const detail: Partial<Record<TerrainLayer, Promise<void>>> = {};
let detailMask: Promise<THREE.DataTexture> | null = null;
let terrainPreview: Promise<TerrainMeta>;
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
    // On the coarse grid: enough for a flat layer
    sea.build(grid, bedTex.userData as never, surfaceTex.userData as never);
    const field = pivotField;
    terrainReady((layer, x, y) => field.smoothAt(layer, x, y));
    eggs.place();
    if (!viewInUrl) {
      // Flying there already (without the heights): the flight goes to the right height instead
      if (submissionBox) frameBox(submissionBox, flight ? 800 : 0);
      else resetView(true);
    }
  }
  textureGrid.bed = textureGrid.surface = grid;
  coverage.bed = bedTex.userData.valid;
  coverage.surface = surfaceTex.userData.valid;
  // Heights are only kept on the CPU for the first (coarse) level
  bedTex.userData = {};
  surfaceTex.userData = {};
  setLayerTextures(bed.material, bedTex, mask, grid);
  setLayerTextures(surface.material, surfaceTex, mask, grid);
  maskTexture?.dispose();
  maskTexture = mask;
  terrainMeta = meta;
  buildMeshes(PREVIEW);
}

/** The detailed height texture of a layer, once. */
function loadDetail(layer: TerrainLayer): Promise<void> {
  detail[layer] ??= (async () => {
    const meta = await terrainPreview;
    const level = meta.resolutions[DETAIL];
    detailMask ??= loadMaskTexture(DATA_URL, level);
    const [height, mask] = await Promise.all([loadHeightTexture(DATA_URL, level, layer, meta.nodata), detailMask]);
    const grid = gridOf(level);
    textureGrid[layer] = grid;
    coverage[layer] = height.userData.valid;
    height.userData = {};
    setLayerTextures((layer === "bed" ? bed : surface).material, height, mask, grid);
    // One mask for both layers, also the one still at its preview heights
    if (maskTexture !== mask) {
      maskTexture?.dispose();
      maskTexture = mask;
      bed.material.uniforms.maskTex.value = surface.material.uniforms.maskTex.value = mask;
    }
    buildMeshes(QUALITY[settings.quality]);
  })();
  return detail[layer]!;
}

/** The terrain layers shown (the velocity lies on the surface), in detail. */
function loadShownTerrain(): Promise<unknown> {
  const shown: TerrainLayer[] = [];
  if (settings.bed) shown.push("bed");
  if (settings.surface || settings.velocity) shown.push("surface");
  return Promise.all(shown.map(loadDetail));
}

/** Bed and surface meshes at a resolution, without triangles where a layer has no data. */
function buildMeshes(res: string): void {
  if (!terrainMeta || !textureGrid.bed || !textureGrid.surface) return;
  const grid = gridOf(terrainMeta.resolutions[res]);
  bedMesh.meshOrigin.value.set(grid.x0, grid.y0);
  bedMesh.meshDx.value = grid.dx;
  const positions = gridPositions(grid);
  bed.geometry.dispose();
  surface.geometry.dispose();
  bed.geometry = gridGeometry(grid, positions, verticesNearData(grid, textureGrid.bed, coverage.bed!));
  surface.geometry = gridGeometry(grid, positions, verticesNearData(grid, textureGrid.surface, coverage.surface!));
  if (velocity) velocity.geometry = surface.geometry;
  bed.visible = settings.bed;
  surface.visible = settings.surface;
  eggs.setIceSurface(settings.surface, settings.surfaceOpacity);
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
  trackBed: false,
  // Layers of the Bedmap survey lines, once a generation is shown
  bedmapTracks: true,
  bedmapBed: false,
  // Bedmap generations shown, by checkbox: none at start
  bedmapDatasets: Object.fromEntries(BEDMAP_DATASETS.map((d) => [d, false])) as Record<string, boolean>,
  iceCores: false,
  groundingLines: false,
  groundingLineScenario: "A" as Scenario,
  // Times shown (ka, 0 for the present), by checkbox
  groundingLineTimes: Object.fromEntries(GROUNDING_LINE_COLORS.map(([age]) => [String(age), true])) as Record<string, boolean>,
  iceDivides: false,
  aircraft: true,
  eggs: true,
  sea: true,
  // Contrast of the ice speed colours: the colours follow log(1 + speed / 10^-contrast m/a)
  velocityContrast: 0,
  lineWidth: 2.5,
  lightAzimuth: 315, // degrees clockwise from grid north (+y)
  lightElevation: 40,
  quality: "Fast",
  lowResWhileMoving: true,
  resetView,
  topView,
  edgeMove: true,
  autoZoom: true,
};

// Private preview of a submission: its isochrones visible through the ice surface, and no
// aircraft, whose animation redraws the close view of the dataset 30 times a second
if (SUBMISSION) {
  settings.aircraft = false;
  settings.surfaceOpacity = 0.5;
  surface.material.uniforms.opacity.value = settings.surfaceOpacity;
  eggs.setIceSurface(settings.surface, settings.surfaceOpacity);
}

// Link preview screenshots (public/og-image.jpg, 1200×630) in development: ?screenshot
// hides the panels (=bedmap adds the Bedmap survey lines); the current image is
// ?screenshot#view=150,-4257,3560,150,-600,150
const SCREENSHOT = import.meta.env.DEV && new URLSearchParams(window.location.search).has("screenshot");
if (SCREENSHOT) {
  document.body.classList.add("clean");
  Object.assign(settings, { surfaceOpacity: 0.6, lineWidth: 3, exaggeration: 100, tracks: false, quality: "Detailed" });
  isochrones.lines.setLineWidth(settings.lineWidth);
  surface.material.uniforms.opacity.value = settings.surfaceOpacity;
  eggs.setIceSurface(settings.surface, settings.surfaceOpacity);
  shared.exaggeration.value = settings.exaggeration;
  for (const layer of lineLayers) layer.setExaggeration(settings.exaggeration);
  iceCores.setExaggeration(settings.exaggeration);
  aircraft.setExaggeration(settings.exaggeration);
  eggs.setExaggeration(settings.exaggeration);
  // ?screenshot=bedmap: with the survey lines of every Bedmap generation (for presentations)
  // thinner, so the isochrones still show through them
  if (new URLSearchParams(window.location.search).get("screenshot") === "bedmap") {
    for (const dataset of BEDMAP_DATASETS) settings.bedmapDatasets[dataset] = true;
    bedmap.tracks.setLineWidth(0.8);
  }
}

function updateBedRange(): void {
  bed.material.uniforms.colormapRange.value.set(-settings.bedRange, settings.bedRange);
  bedmap.setBedRange(settings.bedRange);
  trackBed.setBedRange(settings.bedRange);
  requestRender();
}

function updateLight(): void {
  const az = THREE.MathUtils.degToRad(settings.lightAzimuth);
  const el = THREE.MathUtils.degToRad(settings.lightElevation);
  shared.lightDir.value.set(Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el));
  requestRender();
}
aircraft.group.visible = settings.aircraft && !strained;
sea.mesh.visible = settings.sea;
if (!SUBMISSION && settings.eggs) sea.setClearing(ENDURANCE_AT[0], ENDURANCE_AT[1], ENDURANCE_CLEARING);
updateLight();
updateBedRange();
applyTheme();

// While the settings of the last visit are put back (at the end of the menu)
let restoringSettings = false;
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
gui
  .add(
    {
      // The tour explains the viewer as it is by default
      tutorial: () => {
        resetSettings();
        tutorial.start();
      },
    },
    "tutorial",
  )
  .name("Tutorial (tour of the viewer)");
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
    if (!restoringSettings) viewToUrl();
    shared.exaggeration.value = v;
    for (const layer of lineLayers) layer.setExaggeration(v);
    iceDivides.setExaggeration(v);
    profileMarker.scale.z = v / 1000;
    inTheWay.setExaggeration(v);
    iceCores.setExaggeration(v);
    aircraft.setExaggeration(v);
    eggs.setExaggeration(v);
    pointerMoved = true;
    requestRender();
  });
view.add(settings, "resetView").name("Reset view");
view.add(settings, "topView").name("Top view (map)");
view.add(settings, "edgeMove").name("Move at the edges of the view").onChange((on: boolean) => {
  if (!on) keyNavigation.setEdge(0, 0);
});
view.add(settings, "autoZoom").name("Fly to a clicked line");
const layers = gui.addFolder("Layers");
layers
  .add(settings, "bed")
  .name("Bed (Bedmap3)")
  .onChange((v: boolean) => {
    bed.visible = v;
    bedRange.show(v);
    if (v) loadDetail("bed").catch(showError);
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
    eggs.setIceSurface(v, settings.surfaceOpacity);
    surfaceOpacity.show(v);
    if (v) loadDetail("surface").catch(showError);
    requestRender();
  });
const surfaceOpacity = layers
  .add(settings, "surfaceOpacity", 0.05, 1, 0.05)
  .name("Surface opacity")
  .onChange((v: number) => {
    surface.material.uniforms.opacity.value = v;
    eggs.setIceSurface(settings.surface, v);
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
velocityOptions
  .add(settings, "velocityContrast", -2, 2, 0.1)
  .name("Contrast (slow ice)")
  .onChange(() => {
    if (velocity) velocity.material.uniforms.speedScale.value = speedScale();
    showVelocity();
  });
velocityOptions.add(settings, "flowLines").name("Flow lines").onChange(showVelocity);
velocityOptions
  .add(settings, "animateFlow")
  .name("Animate flow")
  .onChange(showAnimated);
velocityOptions.hide();
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

function showBedmap(): void {
  const shown = BEDMAP_DATASETS.filter((d) => settings.bedmapDatasets[d]);
  // Its layers only once a generation is chosen
  bedmapLayers?.show(shown.length > 0);
  const tracksOn = settings.bedmapTracks && shown.length > 0;
  const bedOn = settings.bedmapBed && shown.length > 0;
  bedmap.tracks.group.visible = tracksOn;
  bedmap.bed.group.visible = bedOn;
  if (tracksOn || bedOn) {
    bedmap
      .load(BEDMAP_URL, shown)
      .then(() => {
        bedmap.setBedRange(settings.bedRange);
        bedmap.applyFilter(new Set(BEDMAP_DATASETS.filter((d) => !settings.bedmapDatasets[d])));
        for (const layer of [bedmap.tracks, bedmap.bed]) layer.invalidate();
        pointerMoved = true;
        requestRender();
      })
      .catch(showError);
  }
  pointerMoved = true;
  requestRender();
}

/** The bed picks of the datasets shown (with the filters), when on. */
function loadTrackBed(): void {
  if (!settings.trackBed) return;
  filtersReady
    .then(() => trackBed.load(isochrones.datasetsNotHidden()))
    .then(() => {
      trackBed.setBedRange(settings.bedRange);
      trackBed.applyFilter(isochrones);
      pointerMoved = true;
      requestRender();
    })
    .catch(showError);
}

function showTrackBed(): void {
  trackBed.group.visible = settings.trackBed;
  loadTrackBed();
  pointerMoved = true;
  requestRender();
}

function showTracks(): void {
  tracks.group.visible = settings.tracks;
  loadTracks();
  pointerMoved = true;
  requestRender();
}
const isochroneFolder = gui.addFolder("Isochrones");
isochroneFolder
  .add(settings, "isochrones")
  .name("Isochrones")
  .onChange((v: boolean) => {
    isochrones.group.visible = v;
    loadIsochrones();
    showIsochroneLegend();
    pointerMoved = true;
    requestRender();
  });
isochroneFolder
  .add(settings, "tracks")
  .name("Surface tracks")
  .onChange(showTracks);
isochroneFolder.add(settings, "trackBed").name("Bed picks").onChange(showTrackBed);
isochroneFolder
  .add(settings, "colorBy", COLOR_BY)
  .name("Colour by")
  .onChange((v: ColorBy) => {
    filtersReady
      .then(() => isochrones.setColorBy(v))
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
    tracks.lines.setLineWidth(Math.max(1, v - 1));
    requestRender();
  });

/** The isochrones of the datasets shown, with the filters. */
function loadIsochrones(): void {
  filtersReady
    .then(() => (settings.isochrones ? isochrones.load(isochrones.shownDatasets()) : undefined))
    .then(() => {
      isochrones.applyFilter();
      showIsochroneLegend();
      pointerMoved = true;
      requestRender();
    })
    .catch(showError);
}

// Filters: datasets and ages shown, filled once the index is loaded
const filters = isochroneFolder.addFolder("Filter");
let filterQueued = false;
/** Apply the filters once, after all the checkboxes changed in this frame. */
function queueFilter(): void {
  if (filterQueued) return;
  filterQueued = true;
  requestAnimationFrame(() => {
    filterQueued = false;
    isochrones.applyFilter();
    tracks.applyFilter(isochrones);
    loadIsochrones();
    loadTracks();
    loadTrackBed();
    for (const layer of lineLayers) layer.invalidate();
    showIsochroneLegend();
    clearHover();
    pointerMoved = true;
    requestRender();
  });
}

/** A folder of boxes, one per value; returns a function to tick the boxes of the values given. */
function checkboxFolder<T>(title: string, items: [label: string, value: T][], hidden: Set<T>): (shown: (value: T) => boolean) => void {
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
  return (shown) => boxes.forEach((box, i) => box.setValue(shown(items[i][1])));
}

/** For matching the names of regions: no case, accents or double spaces. */
/** A "?" after the name of a controller, its explanation shown on hover (or focus, by keyboard). */
function addHelp(controller: { $name: HTMLElement }, text: string): void {
  const help = document.createElement("span");
  help.className = "gui-help";
  help.textContent = "?";
  help.tabIndex = 0;
  help.dataset.help = text;
  help.setAttribute("aria-label", text);
  // Not a click on the box of the controller
  help.addEventListener("click", (event) => event.preventDefault());
  controller.$name.append(help);
}

/**
 * Regions: ice sheets, IMBIE basins, or places of the SCAR Composite Gazetteer (lines within a
 * radius), typed or picked from the suggestions by any of their names (anta_database.regions).
 * Each region chosen is listed (× to remove it), and the field stays to add more. With regions,
 * only their lines are shown, and in the Datasets folder only the datasets with lines there are
 * ticked; "Whole datasets" shows every line of those datasets. All regions removes them all and
 * ticks every dataset again. The regions chosen are kept with the settings (a hidden controller).
 * showDatasets: ticks the boxes of the Datasets folder (null in a private preview, without it)
 */
function regionFilter(index: IrhIndex, regions: Region[], showDatasets: ((shown: (dataset: string) => boolean) => void) | null): void {
  const label = (r: Region) => (r.key ? `${r.name} (${r.key})` : r.name);
  const describe = (r: Region) =>
    r.kind === "place" ? `${r.type ?? "Place"}, lines within ${r.radius_km} km` : r.kind === "basin" ? "IMBIE basin" : "Ice sheet";
  const datasetsOf = (r: Region) => new Set(r.tracks.map((t) => index.tracks[t].dataset));
  const count = (n: number, what: string) => `${n} ${what}${n > 1 ? "s" : ""}`;
  const byName = new Map<string, Region>();
  for (const r of regions) byName.set(normalise(label(r)), r);
  for (const r of regions) {
    for (const name of [r.key, r.name, ...r.aliases]) if (name && !byName.has(normalise(name))) byName.set(normalise(name), r);
  }
  // Suggestions: our own list, not a <datalist>, which Firefox filters only from the start of the
  // text (no "Pine" in "Amundsen Sea Embayment (Pine Island, Thwaites)") and does not show aliases
  const suggest = regionSearch(regions, label);

  const folder = filters.addFolder("Regions").close();
  // chosen: the labels of the regions, "|" between them (saved with the settings)
  const state = { chosen: "", wholeDatasets: false };
  let chosen: Region[] = [];
  const items = document.createElement("ul");
  items.className = "region-list";
  const field = document.createElement("input");
  field.type = "text";
  field.className = "region-search";
  field.autocomplete = "off";
  field.setAttribute("role", "combobox");
  field.setAttribute("aria-autocomplete", "list");
  field.setAttribute("aria-controls", "region-suggestions");
  field.placeholder = "Add a region: Dome C, WSB, Dronning Maud Land...";
  field.setAttribute("aria-label", "Add a region");
  const suggestions = document.createElement("ul");
  suggestions.className = "region-suggest";
  suggestions.id = "region-suggestions";
  suggestions.setAttribute("role", "listbox");
  const info = document.createElement("div");
  info.className = "region-info";

  const showLines = () => {
    isochrones.regionLines =
      chosen.length && !state.wholeDatasets
        ? new Set(chosen.flatMap((r) => r.tracks.map((t) => lineKey(index.tracks[t].dataset, index.tracks[t].line_id))))
        : null;
    queueFilter();
  };
  const showList = () => {
    items.replaceChildren(
      ...chosen.map((r) => {
        const item = document.createElement("li");
        const name = document.createElement("span");
        name.textContent = label(r);
        name.title = `${describe(r)}: ${count(datasetsOf(r).size, "dataset")}, ${count(r.tracks.length, "line")}. Click to fly there`;
        name.addEventListener("click", () => frameBox(r.box, 1500));
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "×";
        remove.title = `Remove ${r.name}`;
        remove.setAttribute("aria-label", `Remove ${r.name}`);
        remove.addEventListener("click", () => setChosen(chosen.filter((c) => c !== r)));
        item.append(name, remove);
        return item;
      }),
    );
    const lines = new Set(chosen.flatMap((r) => r.tracks)).size;
    const datasets = new Set(chosen.flatMap((r) => [...datasetsOf(r)])).size;
    info.textContent = chosen.length
      ? `${count(datasets, "dataset")}, ${count(lines, "line")} in ${chosen.length > 1 ? "these regions" : "this region"}`
      : "All the datasets. Add an ice sheet, an IMBIE basin or a place of the SCAR gazetteer";
  };
  /** The regions chosen; restoring: from the settings, the boxes of the datasets left as they were. */
  const setChosen = (next: Region[], restoring = false) => {
    chosen = next;
    showList();
    showLines();
    if (restoring) return;
    saved.setValue(chosen.map(label).join("|")); // kept with the settings
    const inside = new Set(chosen.flatMap((r) => [...datasetsOf(r)]));
    showDatasets?.((dataset) => chosen.length === 0 || inside.has(dataset));
  };

  const saved = folder
    .add(state, "chosen")
    .onChange((value: string) => {
      if (!restoringSettings) return;
      const restored = value.split("|").map((name) => byName.get(normalise(name)));
      setChosen(restored.filter((r): r is Region => r !== undefined), true);
    })
    .hide();
  let shown: Region[] = [];
  let active = 0; // the suggestion picked by Enter, moved with the arrows
  const showSuggestions = () => {
    const found = suggest(field.value);
    shown = found.map((f) => f.region);
    active = Math.min(active, Math.max(shown.length - 1, 0));
    suggestions.replaceChildren(
      ...found.map(({ region: r, matched }, i) => {
        const item = document.createElement("li");
        item.setAttribute("role", "option");
        item.setAttribute("aria-selected", String(i === active));
        if (i === active) item.className = "active";
        const name = document.createElement("span");
        name.textContent = label(r);
        const about = document.createElement("small");
        about.textContent = `${matched ? `"${matched}", ` : ""}${describe(r)}: ${count(datasetsOf(r).size, "dataset")}, ${count(r.tracks.length, "line")}`;
        item.append(name, about);
        // mousedown, not click: before the field loses the focus
        item.addEventListener("mousedown", (event) => {
          event.preventDefault();
          add(r);
        });
        return item;
      }),
    );
    if (field.value.trim() && !found.length) {
      const none = document.createElement("li");
      none.className = "none";
      none.textContent = `No line in a region called "${field.value.trim()}"`;
      suggestions.append(none);
    }
  };
  const add = (region: Region) => {
    field.value = "";
    active = 0;
    showSuggestions();
    if (!chosen.includes(region)) setChosen([...chosen, region]);
    frameBox(region.box, 1500);
  };
  field.addEventListener("keydown", (event) => {
    event.stopPropagation(); // typing, not moving the view
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (shown.length) active = (active + (event.key === "ArrowDown" ? 1 : shown.length - 1)) % shown.length;
      showSuggestions();
    } else if (event.key === "Escape") {
      field.value = "";
      showSuggestions();
    } else if (event.key === "Enter") {
      // The exact name first (also a name not among the first suggestions), else the one picked
      const exact = byName.get(normalise(field.value));
      const region = exact ?? shown[active];
      if (region) add(region);
    }
  });
  field.addEventListener("input", () => {
    active = 0;
    showSuggestions();
  });
  field.addEventListener("focus", showSuggestions);
  field.addEventListener("blur", () => suggestions.replaceChildren());
  folder.$children.append(items, field, suggestions, info);
  const whole = folder
    .add(state, "wholeDatasets")
    .name("Whole datasets")
    .onChange(showLines);
  addHelp(
    whole,
    "Off: only the lines that cross the regions chosen are shown. On: every line of the datasets that cross them, also outside the regions.",
  );
  folder.add({ all: () => setChosen([]) }, "all").name("All regions");
  showList();
}

function buildFilters(index: IrhIndex, regions: Region[] | null): void {
  let showDatasets: ((shown: (dataset: string) => boolean) => void) | null = null;
  // Datasets by their citation, e.g. "Cavitte et al. (2020)"; not in a private preview, which has
  // only the submitted dataset
  if (!SUBMISSION) {
    const labels = new Map<string, string>();
    for (const f of index.features) if (!labels.has(f.dataset)) labels.set(f.dataset, f.citation ?? f.dataset);
    const datasets = [...labels].sort((a, b) => a[1].localeCompare(b[1])).map(([name, label]): [string, string] => [label, name]);
    showDatasets = checkboxFolder("Datasets", datasets, isochrones.hiddenDatasets);
  }
  // The regions, setting the boxes of the datasets
  if (regions?.length) regionFilter(index, regions, showDatasets);
  // The ages, then the isochrones without age, then the basal unit (not an age, here for now)
  const rank = (k: AgeKey) => (typeof k === "number" ? 0 : k === null ? 1 : 2);
  const ages = [...new Set(index.features.map(ageKey))].sort((a, b) => rank(a) - rank(b) || (a as number) - (b as number));
  checkboxFolder(
    "Ages",
    ages.map((a): [string, AgeKey] => [a === "basal unit" ? "Basal unit" : a === null ? "No age" : `${a} ka`, a]),
    isochrones.hiddenAges,
  );
}

// Bedmap survey lines, traced or not: not in a private preview, which only shows its dataset
// The generations, then their layers, only shown once one is chosen
const bedmapFolder = SUBMISSION ? null : gui.addFolder("Bedmap survey lines");
const bedmapLayers = bedmapFolder?.addFolder("Layers") ?? null;
if (bedmapFolder && bedmapLayers) {
  for (const dataset of BEDMAP_DATASETS) bedmapFolder.add(settings.bedmapDatasets, dataset).onChange(showBedmap);
  // After the generations in the folder
  bedmapFolder.$children.appendChild(bedmapLayers.domElement);
  bedmapLayers.add(settings, "bedmapTracks").name("Surface tracks").onChange(showBedmap);
  bedmapLayers.add(settings, "bedmapBed").name("Bed picks").onChange(showBedmap);
  bedmapLayers.hide();
  bedmapFolder.close();
}

// Not in a private preview, which has no aircraft (above) and no eggs
const miscFolder = SUBMISSION ? null : gui.addFolder("Miscellaneous");
if (miscFolder) {
  miscFolder
    .add(settings, "aircraft")
    .name("Survey aircraft")
    .onChange((v: boolean) => {
      showAnimated();
      // The tracks are their flight paths
      if (v) loadTracks();
    });
  miscFolder
    .add(settings, "sea")
    .name("Sea level")
    .onChange((v: boolean) => {
      sea.mesh.visible = v;
      lastSeaUpdate = performance.now();
      requestRender();
    });
  miscFolder.add(settings, "eggs").name("Hidden eggs").onChange(showEggs);
  miscFolder.close();
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
    if (Object.values(textureGrid).some((grid) => grid.dx < 4)) buildMeshes(QUALITY[v]);
  });
performanceFolder.add(settings, "lowResWhileMoving").name("Lower resolution while moving");
performanceFolder.close();

// The settings of the last visit, kept in the browser (localStorage, by the names in the menu)
// and restored by the menu, as if chosen again. The first visit has the defaults: an almost
// opaque ice surface, the isochrones found by exploring. Not in a private preview, which has
// its own, nor saved from it. The filters are restored once built, with the data
const SETTINGS_KEY = "viewer-settings";
const REMEMBER = !SUBMISSION && !SCREENSHOT && !LESSON;
interface SavedGui {
  controllers: Record<string, unknown>;
  folders: Record<string, SavedGui>;
}
function storedSettings(): SavedGui | null {
  try {
    return REMEMBER ? (JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "null") as SavedGui | null) : null;
  } catch {
    return null;
  }
}
const lastSettings = storedSettings();
// Before 2026-11-12, the Bedmap generations were all ticked and the layers off: shown by the
// layers. Now the generations show them: ticked only if a layer was on
const oldBedmap = lastSettings?.folders["Bedmap survey lines"];
if (oldBedmap && "Measured bed" in oldBedmap.controllers) {
  const { "Surface tracks": tracksOn, "Measured bed": bedOn, ...generations } = oldBedmap.controllers;
  oldBedmap.controllers = generations;
  if (!tracksOn && !bedOn) for (const dataset of BEDMAP_DATASETS) oldBedmap.controllers[dataset] = false;
}
// The filters of the last visit: in the Isochrones folder, or at the top before (2026-11-09)
const lastFilters = lastSettings?.folders.Isochrones?.folders.Filter ?? lastSettings?.folders.Filter;
let filtersBuilt = false;
function saveSettings(): void {
  if (!REMEMBER || restoringSettings) return;
  try {
    const saved = gui.save() as SavedGui;
    // The filters of the last visit, until they are in the menu
    if (!filtersBuilt && lastFilters) saved.folders.Isochrones.folders.Filter = lastFilters;
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(saved));
  } catch {
    // storage blocked, or two entries of the same name: nothing kept
  }
}
function restoreSettings(folder: GUI, saved: SavedGui | undefined): void {
  if (!saved) return;
  restoringSettings = true;
  try {
    folder.load(saved);
  } finally {
    restoringSettings = false;
  }
}
gui.onChange(saveSettings);
function resetSettings(): void {
  gui.reset();
  try {
    localStorage.removeItem(SETTINGS_KEY);
  } catch {
    // storage blocked
  }
  tourView();
}
gui.add({ reset: resetSettings }, "reset").name("Reset all settings");

// Tour of the viewer, started by itself on the first visit (not in a private preview)
const openInMenu = (folder: GUI) => () => {
  setSidebar(true);
  // With the folders it is in
  for (let f: GUI | undefined = folder; f; f = f.parent) f.open();
  folder.domElement.scrollIntoView({ block: "nearest" });
};
// The view the tour comes back to: the whole continent, or the dataset of a private preview
const tourView = () => (SUBMISSION && submissionBox ? frameBox(submissionBox, 1000) : resetView());
const tutorial = new Tutorial([
  {
    title: SUBMISSION ? "Your dataset in the 3D viewer" : "Welcome to the AntADatabase 3D viewer",
    text:
      (SUBMISSION
        ? "The lines under the ice surface are the **radar isochrones** of your submission, as the AntADatabase " +
          "compiled them: layers inside the ice sheet that were once the surface, all of the same age. "
        : "The lines under the ice surface are **radar isochrones**: layers inside the ice sheet that were " +
          "once the surface, all of the same age, traced by radar surveys. ") +
      "Below them, the bed of Antarctica (Bedmap3). A short tour of the viewer?",
    prepare: tourView,
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
    title: "View",
    text: "Stretch the heights with the vertical exaggeration, or look from the top as on a map.",
    tip: "Lost in the viewer? Reset view brings you back to the whole continent.",
    target: () => view.domElement,
    prepare: openInMenu(view),
  },
  {
    title: "Layers",
    text:
      "Show or hide the ice surface, the bed, the ice velocity, the grounding lines since the last ice age, the ice " +
      "divides and the deep ice cores.",
    tip: "Decrease the surface opacity to discover the isochrones under the ice!",
    target: () => layers.domElement,
    prepare: () => {
      tourView();
      openInMenu(layers)();
    },
  },
  {
    title: "Point and click",
    text:
      "**Hover** a line for its details: its dataset, age and depth (on a touch screen, **press and hold** it). " +
      "**Click** an isochrone to fly to its radar line and open its stratigraphy: all its isochrones, along the " +
      "line, over the bed.",
  },
  {
    title: "Colours",
    text: "The isochrones are coloured by their age, in thousands of years (ka). The legend is here; the menu can also colour them by depth, fractional depth or elevation.",
    target: () => document.querySelector("#legends .legend:not([hidden])"),
  },
  {
    title: "Filter",
    text: SUBMISSION
      ? "Choose the ages shown, to look at your isochrones one by one."
      : "Choose the datasets and the ages shown. Every dataset can be downloaded from the Download page, with how to cite it.",
    target: () => filters.domElement,
    prepare: () => {
      tourView();
      openInMenu(filters)();
    },
  },
  // Not in a private preview, which only shows its dataset
  ...(bedmapFolder
    ? [
        {
          title: "Bedmap survey lines",
          text:
            "The radar data of Antarctica go far beyond the isochrones traced so far. Show the **Bedmap survey " +
            "lines** to see where radar surveys flew, and the bed they measured along their way: the lines with no " +
            "isochrone yet are radar data still waiting to be traced.",
          target: () => bedmapFolder.domElement,
          prepare: () => {
            tourView();
            openInMenu(bedmapFolder)();
          },
        },
      ]
    : []),
  {
    title: "Let's go!",
    text: SUBMISSION
      ? "That's all. Explore your isochrones, and see how they fit with the bed and the other layers."
      : "That's all. Now fly over Antarctica, dive under the ice and explore its layers: there is a lot to discover.",
    tip: "The tour can be found again in the menu, under Tutorial.",
    prepare: tourView,
  },
]);

// Start: quick preview, then the detailed terrain of the layers shown
const loading = document.getElementById("loading")!;
filtersReady = irhIndex.then(async (index) => {
  isochrones.init(IRH_URL, index);
  tracks.init(IRH_URL, index);
  trackBed.init(IRH_URL, index);
  // Not in a private preview, which has only its dataset
  const regions = index.regions && !SUBMISSION ? await loadRegions(IRH_URL, index.regions).catch((error) => {
        console.error("Regions of the filter:", error);
        return null;
      }) : null;
  buildFilters(index, regions);
  filtersBuilt = true;
  restoreSettings(filters, lastFilters);
});
terrainPreview = loadMeta(DATA_URL).then(async (meta) => {
  await loadLevel(meta, PREVIEW);
  return meta;
});
restoreSettings(gui, lastSettings ?? undefined);
if (!viewFromUrl()) resetView(true);
terrainPreview
  .then(async () => {
    if (!SUBMISSION && !LESSON && !tutorialSeen() && !document.body.classList.contains("clean")) tutorial.start();
    // A private preview asks instead, once its data is shown (often after its compilation)
    if (SUBMISSION) irhIndex.then(() => askForTour(() => tutorial.start())).catch(() => {});
    loading.textContent = "Loading detailed terrain…";
    await loadShownTerrain();
    loading.remove();
  })
  .catch((error) => {
    loading.textContent = String(error.message ?? error);
    console.error(error);
  });

showTracks();
showBedmap();
loadIsochrones();

// Debugging from the browser console in development: window.viewer
if (LESSON) {
  const settingsControl = (property: string) => gui.controllersRecursive().find((c) => c.property === property);
  import("./lesson")
    .then(({ setupLesson }) =>
      setupLesson({
        camera,
        heightScale: () => shared.exaggeration.value / 1000,
        index: irhIndex,
        irhUrl: IRH_URL,
        ageColor: (feature) => isochrones.ageColor(feature),
        fly: (target, position, ms, orbit) => new Promise((resolve) => flyTo(target, position, ms, () => resolve(), orbit, true)),
        select: selectLine,
        alone: (track) => {
          if (track) {
            isochrones.lines.setHidden((f) => isochrones.features[f]?.line_id !== track.line_id);
            tracks.lines.setHidden((t) => tracks.tracks[t]?.line_id !== track.line_id);
          } else {
            isochrones.applyFilter();
            tracks.applyFilter(isochrones);
          }
          requestRender();
        },
        setSidebar,
        get: (property) => settingsControl(property)?.getValue(),
        set: (property, value) => settingsControl(property)?.setValue(value),
        home: () => resetView(),
      }),
    )
    .catch(showError);
}
// Local development only: the last moves of the camera, sent to the dev server with F9
// (vite.config.ts writes them to /tmp/anta-view-debug.json)
if (import.meta.env.DEV) {
  const history: unknown[] = [];
  const round = (v: { toArray(): number[] }) => v.toArray().map((x) => +x.toFixed(3));
  const record = (type: string, extra: object = {}) => {
    history.push({
      t: Math.round(performance.now()),
      type,
      camera: round(camera.position),
      quaternion: round(camera.quaternion),
      target: round(controls.target),
      distance: +camera.position.distanceTo(controls.target).toFixed(3),
      minDistance: controls.minDistance,
      angleToTarget: +THREE.MathUtils.radToDeg(camera.getWorldDirection(new THREE.Vector3()).angleTo(controls.target.clone().sub(camera.position))).toFixed(3),
      flight: flight !== null,
      keys: keyNavigation.active,
      ...extra,
    });
    if (history.length > 2000) history.splice(0, history.length - 2000);
  };
  let lastChange = 0;
  controls.addEventListener("start", () => record("start"));
  controls.addEventListener("end", () => record("end"));
  controls.addEventListener("change", () => {
    if (performance.now() - lastChange > 100) record("change");
    lastChange = performance.now();
  });
  renderer.domElement.addEventListener("wheel", (e) => record("wheel", { x: e.clientX, y: e.clientY, deltaY: e.deltaY }), { capture: true });
  renderer.domElement.addEventListener("pointerdown", (e) => record("pointerdown", { x: e.clientX, y: e.clientY, button: e.button }), { capture: true });
  window.addEventListener("keydown", (e) => {
    if (e.key !== "F9") return;
    record("F9", { exaggeration: shared.exaggeration.value, size: [window.innerWidth, window.innerHeight], hash: location.hash });
    fetch("/__view-debug", { method: "POST", body: JSON.stringify(history) }).then(() => console.log("View history sent"));
  });
}
if (import.meta.env.DEV) Object.assign(window, { viewer: { scene, camera, controls, keyNavigation, isochrones, tracks, iceCores, aircraft, renderer, bed, surface, settings, buildMeshes, groundingLines, iceDivides, profile, openProfileAt, tutorial, eggs, eggCounter, sea } });
