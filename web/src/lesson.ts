/**
 * Page for teaching (/lesson/): the 3D viewer, with a button that flies to a radar line, turns to
 * see it from the side, and fades into a 2D cross section of the ice with an animation of how
 * its layers formed: snow falls on the surface, each layer is buried and sinks, and the ice flows.
 *
 * An illustration, not a model: the ice sheet is taken as steady. A layer of age a, at a time T
 * ago, lies where the isochrone of age a - T lies today, interpolated between the isochrones in
 * each column of the profile. It starts at the surface (age 0) and ends where the radar sees it.
 * The arrows are schematic: downhill along the surface, faster near the surface (shallow ice).
 */
import * as THREE from "three";
import "./lesson.css";
import { fetchBytes } from "./terrain";

const DATASET = "Cavitte_2020";
const LINE = "VCD_JKB2g_DVD01a";
const COLUMNS = 700; // of the cross section
const MIN_COVERAGE = 0.7; // of the line, for a layer to be used: the gaps are interpolated
const DURATION = 60; // s, at normal speed
const END_HOLD = 4; // s at the end of the animation, showing the present
const FAINT_LAYERS = 60; // between the isochrones, interpolated
const PLOT_WIDTH = 0.86; // of the window
const PLOT_HEIGHT = 0.68;
const SKY = 0.16; // of the plot above the highest surface
const FIRN_DEPTH = 100; // m, drawn as a white band
const BUBBLES = 1000;
const STREAKS = 350;
const FLOW_RATE = 7; // columns per second, of the fastest ice: as the arrows

interface Feature {
  age_ka?: number | null;
}
interface Track {
  dataset: string;
  line_id: string;
  citation: string | null;
  surface: string;
  profile?: { file: string; layers: number[] };
}

export interface LessonViewer {
  camera: THREE.PerspectiveCamera;
  heightScale(): number; // km in the scene per m
  index: Promise<{ features: Feature[]; tracks: Track[] }>;
  irhUrl: string;
  ageColor(feature: number): string;
  fly(target: THREE.Vector3, position: THREE.Vector3, ms: number, orbit: boolean): Promise<void>;
  select(track: Track | null): void;
  /** Show only this line, or all again. */
  alone(track: Track | null): void;
  setSidebar(open: boolean): void;
  get(property: string): unknown;
  set(property: string, value: unknown): void;
  home(): void;
}

interface Section {
  length: number; // km
  distance: Float32Array; // km, of each column
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  surface: Float32Array; // m
  thickness: Float32Array; // m
  ages: number[]; // ka, of the layers, youngest first
  colors: string[];
  downhill: number; // 1 when the surface goes down along the distance, or -1
  /** Depth (m) in a column of the isochrone of an age (ka): per column, on AGE_NODES ages */
  table: Float32Array;
  oldest: number; // ka
}
const AGE_NODES = 400;

/** Monotone cubic interpolation (Fritsch-Carlson), for values at increasing x, at xs. */
function monotone(x: number[], y: number[], xs: number[]): number[] {
  const n = x.length;
  const slopes = x.slice(0, -1).map((v, i) => (y[i + 1] - y[i]) / (x[i + 1] - v));
  const m = y.map((_, i) => {
    if (i === 0) return slopes[0];
    if (i === n - 1) return slopes[n - 2];
    return slopes[i - 1] * slopes[i] <= 0 ? 0 : (2 * slopes[i - 1] * slopes[i]) / (slopes[i - 1] + slopes[i]);
  });
  let k = 0;
  return xs.map((v) => {
    while (k < n - 2 && v > x[k + 1]) k++;
    const h = x[k + 1] - x[k];
    const t = Math.min(1, Math.max(0, (v - x[k]) / h));
    const [t2, t3] = [t * t, t * t * t];
    return (2 * t3 - 3 * t2 + 1) * y[k] + (t3 - 2 * t2 + t) * h * m[k] + (-2 * t3 + 3 * t2) * y[k + 1] + (t3 - t2) * h * m[k + 1];
  });
}

/** Fill the gaps of values along a column index, linearly; constant beyond the ends. */
function fillGaps(values: Float32Array): void {
  let last = -1;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) continue;
    if (last < 0) values.fill(values[i], 0, i);
    else for (let j = last + 1; j < i; j++) values[j] = values[last] + ((values[i] - values[last]) * (j - last)) / (i - last);
    last = i;
  }
  if (last >= 0) values.fill(values[last], last + 1);
}

async function loadSection(viewer: LessonViewer, track: Track, features: Feature[]): Promise<Section> {
  const { file, layers } = track.profile!;
  const rows = new Float32Array(await fetchBytes(`${viewer.irhUrl}/${file}`));
  const width = 6 + layers.length;
  const n = rows.length / width;
  const valid: number[] = [];
  for (let r = 0; r < n; r++) if (Number.isFinite(rows[r * width])) valid.push(r);
  const first = valid[0];
  const last = valid[valid.length - 1];
  const start = rows[first * width + 2];
  const length = rows[last * width + 2] - start;
  const column = (r: number) => Math.min(COLUMNS - 1, Math.floor(((rows[r * width + 2] - start) / length) * COLUMNS));

  // Means per column: rows are much denser than columns
  const mean = (offset: number) => {
    const sum = new Float64Array(COLUMNS);
    const count = new Uint32Array(COLUMNS);
    for (const r of valid) {
      const v = rows[r * width + offset];
      if (Number.isFinite(v)) {
        sum[column(r)] += v;
        count[column(r)]++;
      }
    }
    return Float32Array.from(sum, (s, c) => (count[c] ? s / count[c] : NaN));
  };
  const surface = mean(3);
  const thickness = mean(5);
  fillGaps(surface);
  fillGaps(thickness);

  // The dated layers covering most of the line, youngest first, in fractions of the thickness
  // to fill their gaps (the layers follow the bed, more than a level)
  const used = layers
    .map((feature, k) => ({ feature, k, age: features[feature]?.age_ka ?? NaN, depth: mean(6 + k) }))
    .filter((l) => Number.isFinite(l.age) && l.depth.filter(Number.isFinite).length >= MIN_COVERAGE * COLUMNS)
    .sort((a, b) => a.age - b.age);
  for (const l of used) {
    const fraction = l.depth.map((d, c) => d / thickness[c]);
    fillGaps(fraction);
    l.depth = fraction.map((f, c) => f * thickness[c]);
  }
  const oldest = used[used.length - 1].age;
  const nodes = Array.from({ length: AGE_NODES }, (_, i) => (i * oldest) / (AGE_NODES - 1));
  const table = new Float32Array(COLUMNS * AGE_NODES);
  const ages = [0, ...used.map((l) => l.age)];
  for (let c = 0; c < COLUMNS; c++) {
    // Layers keep their order: noisy picks must not cross
    let deepest = 0;
    const depths = [0, ...used.map((l) => (deepest = Math.min(thickness[c], Math.max(deepest + 0.5, l.depth[c]))))];
    table.set(monotone(ages, depths, nodes), c * AGE_NODES);
  }

  // One way for the arrows: down the surface from one end of the line to the other
  const part = Math.round(COLUMNS / 10);
  const sum = (from: number) => surface.subarray(from, from + part).reduce((total, v) => total + v, 0);
  const downhill = sum(0) >= sum(COLUMNS - part) ? 1 : -1;

  const point = (r: number) => [rows[r * width], rows[r * width + 1]];
  const [x0, y0] = point(first);
  const [x1, y1] = point(last);
  return {
    length,
    distance: Float32Array.from({ length: COLUMNS }, (_, c) => ((c + 0.5) / COLUMNS) * length),
    x0,
    y0,
    x1,
    y1,
    surface,
    thickness,
    ages: used.map((l) => l.age),
    colors: used.map((l) => viewer.ageColor(l.feature)),
    downhill,
    table,
    oldest,
  };
}

/** Depth (m) in a column of the isochrone of an age (ka). */
function depthAt(s: Section, column: number, age: number): number {
  const f = (Math.max(0, age) / s.oldest) * (AGE_NODES - 1);
  const i = Math.min(AGE_NODES - 2, Math.floor(f));
  const a = s.table[column * AGE_NODES + i];
  return a + (s.table[column * AGE_NODES + i + 1] - a) * (f - i);
}

/** Value between two columns. */
function between(values: Float32Array, c: number): number {
  const i = Math.min(COLUMNS - 2, Math.floor(c));
  return values[i] + (values[i + 1] - values[i]) * (c - i);
}

/** A repeatable number in 0-1, for the snowflakes, the arrows and the bubbles. */
const random = (i: number) => {
  const v = Math.sin(i * 127.1 + 11.7) * 43758.5453;
  return v - Math.floor(v);
};

const CAPTIONS: [number, string][] = [
  [0, "Snow falls on the surface of the ice sheet, year after year."],
  [0.14, "Each snowfall is a layer. Buried by the next ones, it compacts: snow, then firn, then ice."],
  [0.36, "Under its own weight, the ice flows slowly downhill, fastest near the surface and almost not at all at the bed."],
  [0.58, "So each layer sinks, and it gets thinner as it spreads out."],
  [0.8, "A layer stays the same age everywhere along the section: it was once the surface. It is an isochrone."],
];
const TODAY_CAPTION = "A cross section of the ice, from the radar: each line is a buried layer, once the surface.";
const END_CAPTION = "Today: the layers the radar sees. Their shapes keep the history of the snow and of the flow of the ice.";

class Stage {
  readonly el = document.createElement("div");
  private canvas = document.createElement("canvas");
  private context = this.canvas.getContext("2d")!;
  private caption = document.createElement("div");
  private clock = document.createElement("div");
  private bar = document.createElement("div");
  private playButton = document.createElement("button");
  private slider = document.createElement("input");
  private arrowsBox = document.createElement("input");
  private speedButtons: HTMLButtonElement[] = [];
  private frame = 0;
  private previous = 0;
  private speed = 1;
  private w = 1;
  private h = 1;
  /** Animation time (s), from 0 to DURATION + END_HOLD; null: the present, before it starts */
  time: number | null = null;
  playing = false;
  // Isochrones of the present fading out when the animation starts
  private fade = 0;

  constructor(
    private s: Section,
    private citation: string,
    private lineId: string,
    onBack: () => void,
  ) {
    this.el.id = "lesson-stage";
    this.caption.id = "lesson-caption";
    this.clock.id = "lesson-clock";
    this.bar.id = "lesson-bar";
    this.playButton.type = "button";
    this.playButton.className = "main";
    this.playButton.addEventListener("click", () => this.toggle());
    this.slider.type = "range";
    this.slider.min = "0";
    this.slider.max = String(DURATION + END_HOLD);
    this.slider.step = "0.05";
    this.slider.addEventListener("input", () => {
      this.pause();
      this.time = Number(this.slider.value);
      this.fade = 0;
      this.draw();
    });
    const restart = this.button("Restart", () => {
      this.time = 0;
      this.fade = 0;
      this.play();
    });
    const speeds = document.createElement("span");
    for (const value of [0.5, 1, 2]) {
      const b = this.button(`×${value}`, () => {
        this.speed = value;
        this.speedButtons.forEach((other) => other.classList.toggle("on", other === b));
      });
      b.classList.toggle("on", value === 1);
      this.speedButtons.push(b);
      speeds.append(b);
    }
    const arrows = document.createElement("label");
    this.arrowsBox.type = "checkbox";
    this.arrowsBox.checked = true;
    this.arrowsBox.addEventListener("change", () => this.draw());
    arrows.append(this.arrowsBox, " Ice flow");
    this.bar.append(this.playButton, restart, this.slider, speeds, arrows, this.button("Back to 3D", onBack));
    this.el.append(this.canvas, this.caption, this.clock, this.bar);
    document.body.append(this.el);
    window.addEventListener("resize", () => this.resize());
    window.addEventListener("keydown", (e) => {
      if (e.code !== "Space" || !this.el.classList.contains("shown")) return;
      e.preventDefault();
      this.toggle();
    });
    this.resize();
  }

  private button(text: string, onClick: () => void): HTMLButtonElement {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.addEventListener("click", onClick);
    return b;
  }

  /** Vertical exaggeration so that the section fills the plot, as a whole number for the viewer. */
  exaggeration(): number {
    const { top, bottom } = this.range();
    const kmPerPixel = this.s.length / (PLOT_WIDTH * window.innerWidth);
    const metres = top - bottom;
    return THREE.MathUtils.clamp(Math.round(((PLOT_HEIGHT * window.innerHeight) / metres) * kmPerPixel * 1000), 1, 100);
  }

  /** Elevations (m) shown: the bed, and sky above the highest surface. */
  range(): { top: number; bottom: number } {
    let [top, bottom] = [-Infinity, Infinity];
    for (let c = 0; c < COLUMNS; c++) {
      top = Math.max(top, this.s.surface[c]);
      bottom = Math.min(bottom, this.s.surface[c] - this.s.thickness[c]);
    }
    const span = (top - bottom) / (1 - SKY);
    return { top: bottom + span, bottom: bottom - 0.04 * span };
  }

  private resize(): void {
    const ratio = Math.min(window.devicePixelRatio, 2);
    [this.w, this.h] = [window.innerWidth, window.innerHeight];
    this.canvas.width = this.w * ratio;
    this.canvas.height = this.h * ratio;
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.draw();
  }

  show(shown: boolean): void {
    this.el.classList.toggle("shown", shown);
    if (!shown) this.pause();
    this.draw();
  }

  /** The present: all isochrones, before the animation. */
  today(): void {
    this.pause();
    this.time = null;
    this.draw();
  }

  toggle(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  play(): void {
    if (this.time === null) this.fade = 1;
    if (this.time === null || this.time >= DURATION + END_HOLD) this.time = 0;
    this.playing = true;
    this.previous = performance.now();
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame((now) => this.tick(now));
    this.updateBar();
  }

  pause(): void {
    this.playing = false;
    cancelAnimationFrame(this.frame);
    this.updateBar();
  }

  private tick(now: number): void {
    const step = Math.min(0.1, (now - this.previous) / 1000);
    this.previous = now;
    this.fade = Math.max(0, this.fade - step / 1.5);
    this.time = (this.time ?? 0) + step * this.speed * (this.fade > 0 ? 0 : 1);
    if (this.time >= DURATION + END_HOLD) {
      this.time = DURATION + END_HOLD;
      this.pause();
    }
    this.draw();
    if (this.playing) this.frame = requestAnimationFrame((t) => this.tick(t));
  }

  private updateBar(): void {
    this.playButton.textContent = this.playing ? "❚❚ Pause" : this.time === null ? "▶ Start the animation" : "▶ Play";
    this.slider.value = String(this.time ?? 0);
  }

  /** Age (ka) of the present moment of the animation: how long ago it is. */
  private ago(): number {
    return this.time === null ? 0 : this.s.oldest * Math.max(0, 1 - this.time / DURATION);
  }

  /** Draw at a time, for recording. */
  seek(time: number, fade = 0): void {
    this.pause();
    this.time = time;
    this.fade = fade;
    this.draw();
  }

  draw(): void {
    const { context: g, s, w, h } = this;
    const { top, bottom } = this.range();
    const left = (w * (1 - PLOT_WIDTH)) / 2;
    const pxPerKm = (w * PLOT_WIDTH) / s.length;
    // As in 3D: the middle of the elevations in the middle of the window
    const pxPerM = (this.exaggeration() * pxPerKm) / 1000;
    const middle = (top + bottom) / 2;
    const X = (c: number) => left + s.distance[c] * pxPerKm;
    const Y = (elevation: number) => h / 2 - (elevation - middle) * pxPerM;
    const start = this.time ?? 0;
    const ago = this.ago();
    const present = this.time === null;

    // Sky, then the ice, white near the surface and bluer with depth
    const sky = g.createLinearGradient(0, 0, 0, Y(top - (top - bottom) * SKY * 0.9));
    sky.addColorStop(0, "#0b1a2c");
    sky.addColorStop(1, "#27486b");
    g.fillStyle = sky;
    g.fillRect(0, 0, w, h);
    const trace = (fromBed: (c: number) => number) => {
      g.beginPath();
      for (let c = 0; c < COLUMNS; c++) g.lineTo(X(c), fromBed(c));
    };
    const surfaceY = (c: number) => Y(s.surface[c]);
    const bedY = (c: number) => Y(s.surface[c] - s.thickness[c]);
    const close = (y: (c: number) => number) => {
      for (let c = COLUMNS - 1; c >= 0; c--) g.lineTo(X(c), y(c));
      g.closePath();
    };
    const ice = g.createLinearGradient(0, Y(top), 0, Y(bottom));
    ice.addColorStop(0, "#f1f8fd");
    ice.addColorStop(0.3, "#bfe0f3");
    ice.addColorStop(0.7, "#74b1dc");
    ice.addColorStop(1, "#2f6ea6");
    trace(surfaceY);
    close(bedY);
    g.fillStyle = ice;
    g.fill();
    // Snow and firn at the top
    for (let k = 4; k >= 1; k--) {
      const depth = (FIRN_DEPTH * k) / 2;
      trace(surfaceY);
      close((c) => Y(s.surface[c] - depth));
      g.fillStyle = "rgba(255, 255, 255, 0.28)";
      g.fill();
    }
    // Bed
    const rock = g.createLinearGradient(0, Y(bottom + (top - bottom) * 0.4), 0, h);
    rock.addColorStop(0, "#8a6a4b");
    rock.addColorStop(1, "#4a3625");
    trace(bedY);
    g.lineTo(X(COLUMNS - 1), h);
    g.lineTo(X(0), h);
    g.closePath();
    g.fillStyle = rock;
    g.fill();

    this.drawIceTexture(start, X, Y);

    // Layers: faint ones between the isochrones, then the isochrones, the oldest first
    const line = (age: number, color: string, width: number, alpha: number) => {
      trace((c) => Y(s.surface[c] - depthAt(s, c, age)));
      g.strokeStyle = color;
      g.lineWidth = width;
      g.globalAlpha = alpha;
      g.stroke();
      g.globalAlpha = 1;
    };
    const visible = present ? 1 : this.fade;
    if (!present || visible > 0) {
      for (let k = 1; k < FAINT_LAYERS; k++) {
        const age = (k * s.oldest) / FAINT_LAYERS - ago;
        if (age > 0) line(age, "#ffffff", 0.7, 0.45);
      }
    }
    const labels: { y: number; text: string; color: string }[] = [];
    for (let i = s.ages.length - 1; i >= 0; i--) {
      if (this.fade > 0) line(s.ages[i], s.colors[i], 1.8, this.fade);
      const age = s.ages[i] - ago;
      if (age < 0 || this.fade > 0) continue;
      line(age, s.colors[i], 1.8, 1);
      labels.push({ y: Y(s.surface[COLUMNS - 1] - depthAt(s, COLUMNS - 1, age)), text: `${Math.round(s.ages[i])} ka`, color: s.colors[i] });
    }

    // Surface
    trace(surfaceY);
    g.strokeStyle = "#ffffff";
    g.lineWidth = 2;
    g.stroke();

    // Ages of the layers, right of the section, as far as they fit
    g.font = "11px system-ui, sans-serif";
    g.textAlign = "left";
    g.textBaseline = "middle";
    let lastY = -Infinity;
    for (const label of labels.sort((a, b) => a.y - b.y)) {
      if (label.y - lastY < 12) continue;
      g.fillStyle = "#dfeaf3";
      g.fillText(label.text, X(COLUMNS - 1) + 7, label.y);
      lastY = label.y;
    }

    if (!present && this.fade === 0) {
      this.drawSnow(start, surfaceY, X);
      if (this.arrowsBox.checked) this.drawArrows(start, X, Y);
    }
    this.drawAxes(pxPerKm, pxPerM, left);

    this.caption.textContent = present
      ? TODAY_CAPTION
      : start >= DURATION
        ? END_CAPTION
        : [...CAPTIONS].reverse().find(([from]) => start / DURATION >= from)![1];
    this.clock.textContent =
      present || start >= DURATION ? "Today" : `${Math.round(ago).toLocaleString("en")},000 years ago`;
    if (!this.slider.matches(":active")) this.slider.value = String(start);
  }

  private drawSnow(time: number, surfaceY: (c: number) => number, X: (c: number) => number): void {
    const g = this.context;
    g.fillStyle = "#ffffff";
    for (let i = 0; i < 260; i++) {
      const c = Math.floor(random(i) * COLUMNS);
      const fall = (time * (0.16 + 0.1 * random(i + 500)) + random(i + 900)) % 1;
      const size = 1 + 1.6 * random(i + 300);
      g.globalAlpha = 0.9;
      g.beginPath();
      g.arc(X(c) + 7 * Math.sin(time * 1.5 + i), fall * surfaceY(c), size, 0, 2 * Math.PI);
      g.fill();
    }
    g.globalAlpha = 1;
  }

  /**
   * Streaks and air bubbles carried by the ice: each keeps a height above the bed, a fraction of
   * the thickness, and sinks a little on its way, so the ice is seen shearing near the bed. Bubbles
   * shrink with depth, as they do under pressure. They leave at one end and come back at the other.
   */
  private drawIceTexture(time: number, X: (c: number) => number, Y: (m: number) => number): void {
    const { context: g, s } = this;
    const place = (c: number) => X(Math.floor(c)) + (X(Math.ceil(c)) - X(Math.floor(c))) * (c - Math.floor(c));
    const carried = (seed: number, shift = 0) => {
      const fraction = 0.015 + 0.975 * random(seed);
      const speed = 1 - (1 - fraction) ** 4;
      const c = (((random(seed + 0.5) * COLUMNS + s.downhill * FLOW_RATE * speed * time + shift) % COLUMNS) + COLUMNS) % COLUMNS;
      const along = s.downhill > 0 ? c / COLUMNS : 1 - c / COLUMNS;
      const height = fraction * (1 - 0.1 * along);
      return { c, height, y: Y(between(s.surface, c) - between(s.thickness, c) * (1 - height)) };
    };

    // Streaks along the flow, faint and of two tones
    g.lineWidth = 1;
    for (const [tone, color] of [
      [0, "rgba(255, 255, 255, 0.32)"],
      [1, "rgba(24, 66, 110, 0.16)"],
    ] as const) {
      g.strokeStyle = color;
      g.beginPath();
      for (let i = 0; i < STREAKS; i++) {
        if (i % 2 !== tone) continue;
        const seed = 2000 + i;
        const a = carried(seed);
        const b = carried(seed, 4 + 16 * random(seed + 0.25));
        if (b.c < a.c) continue; // across the ends of the section
        g.moveTo(place(a.c), a.y);
        g.lineTo(place(b.c), a.y); // level: following the bed, deep ones would look like scratches
      }
      g.stroke();
    }

    // Bubbles, from small and faint near the bed to larger near the surface
    for (const [low, radius, alpha] of [
      [0, 0.8, 0.4],
      [0.35, 1.2, 0.6],
      [0.7, 1.8, 0.8],
    ]) {
      g.fillStyle = `rgba(255, 255, 255, ${alpha})`;
      g.beginPath();
      for (let i = 0; i < BUBBLES; i++) {
        const { c, height, y } = carried(5000 + i);
        if (height < low || height >= low + 0.35) continue;
        const r = radius * (0.7 + 0.6 * random(i + 0.75));
        g.moveTo(place(c) + r, y);
        g.arc(place(c), y, r, 0, 2 * Math.PI);
      }
      g.fill();
    }
  }

  /** Arrows moving along levels of the ice: a level stays at a fraction of the thickness. */
  private drawArrows(time: number, X: (c: number) => number, Y: (m: number) => number): void {
    const { context: g, s } = this;
    const levels = 9; // heights above the bed, as fractions of the thickness
    const across = 24;
    g.strokeStyle = g.fillStyle = "#ff7a1a";
    g.lineWidth = 1.5;
    for (let level = 0; level < levels; level++) {
      const fraction = (level + 0.6) / levels;
      const speed = 1 - (1 - fraction) ** 4; // shallow ice: the shear is near the bed
      for (let a = 0; a < across; a++) {
        const home = ((a + 0.5 + 0.5 * (level % 2)) / across) * (COLUMNS - 1);
        // Each arrow moves over a short way, then starts again, fading at both ends, and sinks a little
        const phase = (time * 0.3 + random(level * 97 + a)) % 1;
        const way = 0.8 * (COLUMNS / across) * speed;
        const at = (t: number) => THREE.MathUtils.clamp(home + s.downhill * t * way, 0, COLUMNS - 1);
        const height = (c: number, t: number) => between(s.surface, c) - between(s.thickness, c) * (1 - fraction * (1 - 0.05 * t));
        const place = (c: number) => X(Math.floor(c)) + (X(Math.ceil(c)) - X(Math.floor(c))) * (c - Math.floor(c));
        const ahead = phase + 0.5;
        const [tail, head] = [at(phase), at(ahead)];
        if (head === 0 || head === COLUMNS - 1) continue; // past the end of the section
        const [x0, y0, x1, y1] = [place(tail), Y(height(tail, phase)), place(head), Y(height(head, ahead))];
        const angle = Math.atan2(y1 - y0, x1 - x0);
        const size = 2.5 + 2.5 * speed;
        g.globalAlpha = 0.85 * Math.sin(Math.PI * phase);
        g.beginPath();
        g.moveTo(x0, y0);
        g.lineTo(x1, y1);
        g.stroke();
        g.beginPath();
        g.moveTo(x1 + size * Math.cos(angle), y1 + size * Math.sin(angle));
        g.lineTo(x1 - 0.5 * size * Math.cos(angle - 1.1), y1 - 0.5 * size * Math.sin(angle - 1.1));
        g.lineTo(x1 - 0.5 * size * Math.cos(angle + 1.1), y1 - 0.5 * size * Math.sin(angle + 1.1));
        g.closePath();
        g.fill();
      }
    }
    g.globalAlpha = 1;
  }

  private drawAxes(pxPerKm: number, pxPerM: number, left: number): void {
    const { context: g, s, h } = this;
    g.fillStyle = "#e8eef4";
    g.font = "12px system-ui, sans-serif";
    g.textBaseline = "alphabetic";
    // A bar for the distance, and one for the height: not to the same scale
    const scale = 50;
    const y = h - 66;
    g.strokeStyle = "#e8eef4";
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(left, y);
    g.lineTo(left + scale * pxPerKm, y);
    g.moveTo(left, y);
    g.lineTo(left, y - 1000 * pxPerM);
    g.stroke();
    g.textAlign = "left";
    g.fillText(`${scale} km`, left + 4, y + 15);
    g.fillText("1 km", left + 6, y - 500 * pxPerM);
    g.textAlign = "right";
    g.fillText(
      `${s.length.toFixed(0)} km long, heights ×${this.exaggeration()}. ${this.citation}, line ${this.lineId}. An illustration: steady ice sheet, schematic flow.`,
      this.w - left,
      y - 6,
    );
  }
}

export async function setupLesson(viewer: LessonViewer): Promise<void> {
  const start = document.createElement("button");
  start.id = "lesson-start";
  start.type = "button";
  start.textContent = "Loading the cross section…";
  start.disabled = true;
  document.body.append(start);

  const { features, tracks } = await viewer.index;
  const track = tracks.find((t) => t.dataset === DATASET && t.line_id === LINE);
  if (!track?.profile) throw new Error(`No profile of ${DATASET} ${LINE}`);
  const section = await loadSection(viewer, track, features);
  start.textContent = "2D animation";
  start.disabled = false;

  // What the flight changes, to put it back
  const CHANGED = ["exaggeration", "surfaceOpacity", "aircraft"] as const;
  let saved: unknown[] = [];

  const back = () => {
    stage.show(false);
    viewer.select(null);
    viewer.alone(null);
    CHANGED.forEach((property, i) => viewer.set(property, saved[i]));
    viewer.setSidebar(true);
    document.body.classList.remove("lesson-flying");
    start.hidden = false;
    viewer.home();
  };
  const stage = new Stage(section, track.citation ?? DATASET, LINE, back);

  const flyIn = async () => {
    start.hidden = true;
    document.body.classList.add("lesson-flying");
    viewer.setSidebar(false);
    saved = CHANGED.map((property) => viewer.get(property));
    viewer.set("aircraft", false);
    viewer.set("surfaceOpacity", 0.3);
    viewer.set("exaggeration", stage.exaggeration());
    viewer.select(track);
    stage.today();

    const { x0, y0, x1, y1, length } = section;
    const heading = Math.atan2(y1 - y0, x1 - x0);
    // The camera on the side from which the distance grows to the right, as in the 2D section
    const [nx, ny] = [Math.sin(heading), -Math.cos(heading)];
    const { top, bottom } = stage.range();
    const scale = viewer.heightScale();
    const target = new THREE.Vector3((x0 + x1) / 2, (y0 + y1) / 2, ((top + bottom) / 2) * scale);
    const camera = viewer.camera;
    const tan = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const distance = length / (PLOT_WIDTH * 2 * tan * camera.aspect);
    const view = (tilt: number) => {
      const t = THREE.MathUtils.degToRad(tilt);
      return target.clone().add(new THREE.Vector3(nx * Math.cos(t), ny * Math.cos(t), Math.sin(t)).multiplyScalar(distance));
    };
    await viewer.fly(target, view(40), 3500, false);
    await new Promise((resolve) => setTimeout(resolve, 500));
    viewer.alone(track);
    // The cross section appears as the view turns level: nothing of the terrain hides it then
    const turn = viewer.fly(target, view(2), 3500, true);
    setTimeout(() => stage.show(true), 2000);
    await turn;
  };
  start.addEventListener("click", () => flyIn().catch(console.error));
  if (import.meta.env.DEV) Object.assign(window, { lesson: { stage, section, flyIn } });
}
