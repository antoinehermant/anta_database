/**
 * Survey aircraft flying along the flight tracks, sending radar pulses down
 * through the ice to the bed: an illustration of how the data is acquired.
 *
 * Sizes, altitude and speed are in pixels, so the aircraft look the same at
 * any zoom. Each aircraft surveys one dataset: it flies the long straight
 * legs of its tracks, and at the end of a leg turns to a nearby one it hasn't
 * flown recently, so it never jumps.
 */
import * as THREE from "three";

// On screen (px): wingspan, height above the surface, speed (px/s)
const WINGSPAN_PX = 26;
const ALTITUDE_PX = 30;
const SPEED_PX = 30;
// Pulses: one every PULSE_INTERVAL s, each going down to the bed in PULSE_LIFE s
const PULSE_INTERVAL = 0.4;
const PULSE_LIFE = 1.6;
const PULSES = Math.ceil(PULSE_LIFE / PULSE_INTERVAL) + 1;
const PULSE_COLOR = new THREE.Color("#d1a10f");

/** Path of a flight: points (x km, y km, surface elevation m) and distance (km) along it. */
export interface FlightPath {
  points: Float32Array;
  distances: Float32Array;
}

/** A straight leg of a track, that the aircraft can fly. */
export interface FlightLine {
  feature: number;
  points: Float32Array; // x km, y km, surface elevation m
  length: number; // km
}

// Legs: where a track bends by more than SIMPLIFY km from a straight line, at
// least MIN_LENGTH km long, or the longest FALLBACK legs of a dataset without such
const SIMPLIFY = 5;
const MIN_LENGTH = 50;
const FALLBACK = 5;

/** Indices of the points kept by a Douglas-Peucker simplification of points (x, y, z rows). */
function corners(points: number[], tolerance: number): number[] {
  const n = points.length / 3;
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const stack: [number, number][] = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop()!;
    const [ax, ay] = [points[3 * a], points[3 * a + 1]];
    const [dx, dy] = [points[3 * b] - ax, points[3 * b + 1] - ay];
    const length = Math.hypot(dx, dy) || 1e-9;
    let worst = -1;
    let distance = tolerance;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs((points[3 * i] - ax) * dy - (points[3 * i + 1] - ay) * dx) / length;
      if (d > distance) [worst, distance] = [i, d];
    }
    if (worst < 0) continue;
    keep[worst] = 1;
    stack.push([a, worst], [worst, b]);
  }
  return [...keep.keys()].filter((i) => keep[i]);
}

/**
 * Straight legs of the tracks of each dataset, from the segments (start xyz,
 * end xyz) of each feature (track) and its dataset.
 */
export function flightLines(
  featureCount: number,
  segmentsOf: (feature: number) => Float32Array | null,
  datasetOf: (feature: number) => string,
): Map<string, FlightLine[]> {
  const legs = new Map<string, FlightLine[]>();
  const addRun = (feature: number, points: number[]) => {
    if (points.length < 6) return;
    const kept = corners(points, SIMPLIFY);
    for (let k = 1; k < kept.length; k++) {
      const leg = new Float32Array(points.slice(3 * kept[k - 1], 3 * kept[k] + 3));
      const n = leg.length / 3;
      const length = Math.hypot(leg[3 * n - 3] - leg[0], leg[3 * n - 2] - leg[1]);
      const dataset = datasetOf(feature);
      if (!legs.has(dataset)) legs.set(dataset, []);
      legs.get(dataset)!.push({ feature, points: leg, length });
    }
  };
  for (let feature = 0; feature < featureCount; feature++) {
    const segments = segmentsOf(feature);
    if (!segments) continue;
    // Runs: where segments join
    let points: number[] = [];
    for (let s = 0; s < segments.length; s += 6) {
      const n = points.length;
      if (n && (points[n - 3] !== segments[s] || points[n - 2] !== segments[s + 1])) {
        addRun(feature, points);
        points = [];
      }
      if (!points.length) points.push(segments[s], segments[s + 1], segments[s + 2]);
      points.push(segments[s + 3], segments[s + 4], segments[s + 5]);
    }
    addRun(feature, points);
  }
  for (const [dataset, all] of legs) {
    all.sort((a, b) => b.length - a.length);
    const long = all.filter((l) => l.length >= MIN_LENGTH);
    legs.set(dataset, long.length >= FALLBACK ? long : all.slice(0, FALLBACK));
  }
  return legs;
}

/** Path through points (x, y, z rows). */
function pathOf(points: Float32Array): FlightPath {
  const distances = new Float32Array(points.length / 3);
  for (let i = 1; i < distances.length; i++) {
    distances[i] = distances[i - 1] + Math.hypot(points[3 * i] - points[3 * i - 3], points[3 * i + 1] - points[3 * i - 2]);
  }
  return { points, distances };
}

/** Points of a line, from its end if reversed, after a point to fly from (or null). */
function route(line: FlightLine, reversed: boolean, from: THREE.Vector3 | null): Float32Array {
  const n = line.points.length / 3;
  const points = new Float32Array((n + (from ? 1 : 0)) * 3);
  let k = 0;
  if (from) points.set([from.x, from.y, from.z], k++ * 3);
  for (let i = 0; i < n; i++, k++) {
    const j = reversed ? n - 1 - i : i;
    points.set(line.points.subarray(3 * j, 3 * j + 3), 3 * k);
  }
  return points;
}

/** Point (x, y, surface m) at a distance along a path. */
function along(path: FlightPath, distance: number, target: THREE.Vector3): THREE.Vector3 {
  const { points, distances } = path;
  let lo = 0;
  let hi = distances.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (distances[mid] <= distance) lo = mid;
    else hi = mid;
  }
  const t = THREE.MathUtils.clamp((distance - distances[lo]) / Math.max(1e-9, distances[hi] - distances[lo]), 0, 1);
  return target.set(
    points[3 * lo] + t * (points[3 * hi] - points[3 * lo]),
    points[3 * lo + 1] + t * (points[3 * hi + 1] - points[3 * lo + 1]),
    points[3 * lo + 2] + t * (points[3 * hi + 2] - points[3 * lo + 2]),
  );
}

// The aircraft are drawn last, opaque, over everything (surface and lines
// included): the depth buffer is cleared just before them, so their parts
// still hide each other correctly. Being "transparent" (with opacity 1) puts
// them in the last pass, ordered by renderOrder.
const AIRCRAFT_ORDER = 99;
const opaqueOnTop = (color: string) => new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 1 });

function depthClear(): THREE.Group {
  const geometry = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(9), 3));
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ transparent: true, colorWrite: false, depthWrite: false, depthTest: false }));
  mesh.frustumCulled = false;
  mesh.onBeforeRender = (renderer) => renderer.clearDepth();
  // The order of a group applies to everything in it
  const group = new THREE.Group();
  group.renderOrder = AIRCRAFT_ORDER - 1;
  return group.add(mesh);
}

/** A small twin-engine aircraft, 1 unit wingspan, nose along +x, z up. */
function aircraftModel(): THREE.Group {
  const body = opaqueOnTop("#f2f2f0");
  const paint = opaqueOnTop("#d8352a");
  const dark = opaqueOnTop("#3a3f47");
  const model = new THREE.Group();
  model.renderOrder = AIRCRAFT_ORDER;
  const add = (geometry: THREE.BufferGeometry, material: THREE.Material, x: number, y = 0, z = 0) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, y, z);
    mesh.frustumCulled = false;
    mesh.renderOrder = AIRCRAFT_ORDER;
    model.add(mesh);
  };
  // Fuselage and nose, along x
  add(new THREE.CylinderGeometry(0.045, 0.03, 0.72, 12).rotateZ(-Math.PI / 2), body, -0.02);
  add(new THREE.SphereGeometry(0.045, 12, 8), body, 0.34);
  add(new THREE.BoxGeometry(0.06, 0.07, 0.03), dark, 0.29, 0, 0.03); // cockpit
  // High wing, engines, tail
  add(new THREE.BoxGeometry(0.13, 1, 0.018), paint, 0.06, 0, 0.04);
  for (const y of [-0.2, 0.2]) add(new THREE.CylinderGeometry(0.028, 0.022, 0.18, 10).rotateZ(-Math.PI / 2), dark, 0.11, y, 0.025);
  add(new THREE.BoxGeometry(0.08, 0.36, 0.012), paint, -0.35, 0, 0.01);
  add(new THREE.BoxGeometry(0.1, 0.014, 0.15), paint, -0.35, 0, 0.08);
  return model;
}

interface Pulse {
  ring: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
  age: number; // s, PULSE_LIFE or more when not used
  x: number;
  y: number;
  top: number; // m, elevation of the aircraft when sent
  bed: number; // m
}

interface Plane {
  model: THREE.Group;
  path: FlightPath | null;
  dataset: string;
  line: FlightLine | null;
  recent: FlightLine[]; // legs flown last
  distance: number; // km along the path
  surface: number; // m, surface elevation under the aircraft
  heading: number; // rad
  sinceLastPulse: number; // s
  pulses: Pulse[];
}

/** Remove the pulses of an aircraft that left: they are no longer animated, and would stay. */
function clearPulses(plane: Plane): void {
  for (const pulse of plane.pulses) {
    pulse.ring.visible = false;
    pulse.age = PULSE_LIFE;
  }
}

export class Aircraft {
  readonly group = new THREE.Group();
  private planes: Plane[] = [];
  private exaggeration = 1;

  /**
   * One aircraft per dataset. lines(dataset) gives the legs to fly (empty
   * until the tracks are loaded), hidden(feature) whether a track is filtered
   * out, and bedAt() the bed elevation (m) at a point, or null.
   */
  constructor(
    datasets: string[],
    private lines: (dataset: string) => FlightLine[],
    private hidden: (feature: number) => boolean,
    private bedAt: (x: number, y: number) => number | null,
  ) {
    const ringGeometry = new THREE.RingGeometry(0.8, 1, 48);
    // Pulses over the transparent surface, after the lines
    const pulses = new THREE.Group();
    pulses.renderOrder = 6;
    this.group.add(pulses, depthClear());
    for (const dataset of datasets) {
      const model = aircraftModel();
      model.visible = false;
      this.group.add(model);
      const plane: Plane = { model, path: null, dataset, line: null, recent: [], surface: 0, distance: 0, heading: 0, sinceLastPulse: 0, pulses: [] };
      for (let k = 0; k < PULSES; k++) {
        const material = new THREE.MeshBasicMaterial({
          color: PULSE_COLOR,
          transparent: true,
          depthWrite: false,
          side: THREE.DoubleSide,
        });
        const ring = new THREE.Mesh(ringGeometry, material);
        ring.visible = false;
        ring.frustumCulled = false;
        pulses.add(ring);
        plane.pulses.push({ ring, age: PULSE_LIFE, x: 0, y: 0, top: 0, bed: 0 });
      }
      this.planes.push(plane);
    }
  }

  setExaggeration(exaggeration: number): void {
    this.exaggeration = exaggeration;
  }

  /** Next path of an aircraft at a position (null when it isn't flying yet). */
  private nextPath(plane: Plane, from: THREE.Vector3 | null): FlightPath | null {
    const lines = this.lines(plane.dataset).filter((l) => !this.hidden(l.feature));
    if (!lines.length) return null;
    if (!from) {
      // First flight: somewhere along any leg
      plane.line = lines[Math.floor(Math.random() * lines.length)];
      plane.heading = NaN;
      return pathOf(route(plane.line, Math.random() < 0.5, null));
    }
    // Then: a leg not flown recently, close to the aircraft and entered from
    // its closest end, preferring small turns; one of the best few, to vary
    const recent = new Set(plane.recent);
    const fresh = lines.filter((l) => !recent.has(l));
    const candidates = fresh.length ? fresh : lines;
    const meanLength = candidates.reduce((sum, l) => sum + l.length, 0) / candidates.length;
    const heading = new THREE.Vector2(Math.cos(plane.heading), Math.sin(plane.heading));
    const here = new THREE.Vector2(from.x, from.y);
    const options: { line: FlightLine; reversed: boolean; cost: number }[] = [];
    for (const line of candidates) {
      const p = line.points;
      const last = p.length / 3 - 1;
      for (const reversed of [false, true]) {
        const [a, b] = reversed ? [last, 0] : [0, last];
        const entry = new THREE.Vector2(p[3 * a], p[3 * a + 1]);
        const direction = new THREE.Vector2(p[3 * b] - p[3 * a], p[3 * b + 1] - p[3 * a + 1]).normalize();
        const turn = (1 - heading.dot(direction)) / 2; // 0 straight on, 1 back
        options.push({ line, reversed, cost: entry.distanceTo(here) + 0.5 * meanLength * turn });
      }
    }
    options.sort((a, b) => a.cost - b.cost);
    const choice = options[Math.floor(Math.random() * Math.min(3, options.length))];
    plane.line = choice.line;
    plane.recent.push(choice.line);
    // Remember about half of the legs, at most 8
    while (plane.recent.length > Math.min(8, Math.floor(lines.length / 2))) plane.recent.shift();
    return pathOf(route(choice.line, choice.reversed, from));
  }

  /** Move by dt seconds, sized for the camera and canvas height (px). */
  update(dt: number, camera: THREE.PerspectiveCamera, height: number): void {
    const z = (m: number) => (m * this.exaggeration) / 1000;
    const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const position = new THREE.Vector3();
    const ahead = new THREE.Vector3();
    for (const plane of this.planes) {
      if (!plane.path) {
        // From where the aircraft is, if it's flying
        const { x, y } = plane.model.position;
        const from = plane.model.visible ? new THREE.Vector3(x, y, plane.surface) : null;
        plane.path = this.nextPath(plane, from);
        if (!plane.path) {
          plane.model.visible = false;
          plane.line = null;
          clearPulses(plane);
          continue;
        }
        // The first flights start somewhere along their leg
        const length = plane.path.distances[plane.path.distances.length - 1];
        plane.distance = from ? 0 : Math.random() * 0.6 * length;
      }
      // A dataset filtered out: its aircraft leaves
      if (plane.line && this.hidden(plane.line.feature)) {
        plane.model.visible = false;
        plane.path = plane.line = null;
        clearPulses(plane);
        continue;
      }
      const path = plane.path;
      along(path, plane.distance, position);
      plane.surface = position.z;
      // Size of a pixel (km) at the aircraft
      const pixel = (2 * tan * camera.position.distanceTo(position.clone().setZ(z(position.z)))) / height;
      const wingspan = WINGSPAN_PX * pixel;
      const altitude = Math.max(z(600), ALTITUDE_PX * pixel);
      const top = position.z + (altitude * 1000) / this.exaggeration; // m

      // Heading towards a point a few wingspans ahead, turning smoothly
      const length = path.distances[path.distances.length - 1];
      along(path, Math.min(length, plane.distance + 3 * wingspan), ahead);
      if (ahead.distanceTo(position) > 1e-6) {
        const target = Math.atan2(ahead.y - position.y, ahead.x - position.x);
        if (Number.isNaN(plane.heading)) plane.heading = target;
        const turn = Math.atan2(Math.sin(target - plane.heading), Math.cos(target - plane.heading));
        plane.heading += turn * Math.min(1, 4 * dt);
      }
      plane.model.visible = true;
      plane.model.position.set(position.x, position.y, z(top));
      plane.model.rotation.set(0, 0, plane.heading);
      plane.model.scale.setScalar(wingspan);

      // Send a pulse now and then; each goes down from the aircraft to the bed, widening and fading
      plane.sinceLastPulse += dt;
      if (plane.sinceLastPulse >= PULSE_INTERVAL) {
        plane.sinceLastPulse = 0;
        const pulse = plane.pulses.reduce((a, b) => (a.age >= b.age ? a : b));
        const bed = this.bedAt(position.x, position.y);
        Object.assign(pulse, { age: 0, x: position.x, y: position.y, top, bed: bed ?? position.z - 2000 });
      }
      for (const pulse of plane.pulses) {
        pulse.age += dt;
        const t = pulse.age / PULSE_LIFE;
        pulse.ring.visible = t < 1;
        if (t >= 1) continue;
        pulse.ring.position.set(pulse.x, pulse.y, z(pulse.top + t * (pulse.bed - pulse.top)));
        pulse.ring.scale.setScalar(wingspan * (0.15 + 1.2 * t));
        pulse.ring.material.opacity = 0.9 * (1 - t) ** 1.5;
      }

      plane.distance += SPEED_PX * pixel * dt;
      if (plane.distance >= length) plane.path = null;
    }
  }
}
