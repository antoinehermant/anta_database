/**
 * Colormaps of the viewer, and scales mapping values to [0, 1].
 *
 * Isochrone colormaps are trimmed from cmastro (MIT, Adrian Price-Whelan):
 * laguna by Peter Karpov, and hesperia. Darker means older (within
 * each part of the age colours), deeper or lower.
 */
import * as THREE from "three";

export class Colormap {
  readonly stops: [number, string][];
  private colors: THREE.Color[];

  /**
   * Css colours, evenly spaced from t = 0 to t = 1, or at given t. Two stops
   * at the same t make a sharp change of colour.
   */
  constructor(stops: string[] | [number, string][]) {
    this.stops = stops.map((s, i) => (typeof s === "string" ? [i / (stops.length - 1), s] : s));
    this.colors = this.stops.map(([, c]) => new THREE.Color(c));
  }

  /** Colour at t (clamped to [0, 1]). */
  at(t: number, target = new THREE.Color()): THREE.Color {
    t = THREE.MathUtils.clamp(Number.isFinite(t) ? t : 0, 0, 1);
    let i = 0;
    while (i < this.stops.length - 2 && t > this.stops[i + 1][0]) i++;
    const [t0] = this.stops[i];
    const [t1] = this.stops[i + 1];
    return target.copy(this.colors[i]).lerp(this.colors[i + 1], t1 > t0 ? (t - t0) / (t1 - t0) : 1);
  }

  css(): string {
    return `linear-gradient(to right, ${this.stops.map(([t, c]) => `${c} ${(t * 100).toFixed(2)}%`).join(", ")})`;
  }
}

/** Colours of stops evenly spaced between t0 and t1. */
function spread(colors: string[], t0: number, t1: number): [number, string][] {
  return colors.map((c, i) => [t0 + ((t1 - t0) * i) / (colors.length - 1), c]);
}

// Hesperia, from light to dark
const HESPERIA = ["#ffeaad", "#fbca7b", "#f4a86c", "#e7866e", "#ce6a7d", "#af5188", "#8d3a8d", "#692689"];
// Laguna, from light to dark
const LAGUNA = ["#bbeecb", "#7ad9b2", "#4bbeaf", "#339fb9", "#337dc0", "#3c5cb8", "#423c9e", "#3f2273", "#301142"];
// Ages: young IRHs (up to AGE_SPLIT_KA) with hesperia over the first
// AGE_SPLIT_T of the colours, older ones with laguna; each from young (light)
// to old (dark)
export const AGE_SPLIT_KA = 9;
export const AGE_SPLIT_T = 0.9;
function ageColormap(splitT: number): Colormap {
  return new Colormap([
    ...spread(HESPERIA, 0, splitT),
    ...spread(LAGUNA, splitT, 1),
  ]);
}
export const AGE = ageColormap(AGE_SPLIT_T);
// The colour bar shows the old part wider, so its labels fit
export const AGE_LEGEND_SPLIT_T = 0.25;
export const AGE_LEGEND = ageColormap(AGE_LEGEND_SPLIT_T);
// Low (dark) to high (light)
export const ELEVATION = new Colormap([...LAGUNA].reverse());
// Shallow (light) to deep (dark), also for fractional depths
export const DEPTH = new Colormap(LAGUNA);
// Ice speed on a log scale: slow (pale) to fast
export const SPEED = new Colormap([
  "#eef4f9", "#c3dcee", "#86b9df", "#4a8ccc", "#4f5fb5", "#7a3f9e", "#b8307a", "#e85b44", "#fca636", "#fbe36a",
]);

/** Bed colours at fractions of the colour range, centred on sea level. */
export const BED_STOPS: [number, string][] = [
  [-1, "#1f4f86"],
  [-0.6, "#3f78b3"],
  [-0.3, "#6fa1cf"],
  [-0.08, "#a3c8e4"],
  [0, "#c9e2ef"],
  [0.0001, "#a9be95"],
  [0.12, "#bfc293"],
  [0.3, "#cdb28d"],
  [0.55, "#ddc9b2"],
  [1, "#f6f3ee"],
];

export interface Scale {
  min: number;
  max: number;
  log: boolean;
  // Log scales only: value at t = splitT (default 0.5), each part being a log scale
  split?: number;
  splitT?: number;
}

/** Position of a value in a scale, in [0, 1] (not clamped). */
export function scaleT(scale: Scale, value: number): number {
  const { min, max, split } = scale;
  if (scale.log && split !== undefined && split > min && split < max) {
    const at = scale.splitT ?? 0.5;
    if (value <= split) return (at * Math.log(value / min)) / Math.log(split / min);
    return at + ((1 - at) * Math.log(value / split)) / Math.log(max / split);
  }
  if (scale.log) return Math.log(value / min) / Math.log(max / min);
  return (value - min) / (max - min);
}

export interface Tick {
  t: number;
  label: string;
  row?: number; // 1 for a second line of labels
}

function format(value: number): string {
  return String(Number(value.toPrecision(Math.abs(value) >= 100 ? 4 : 2)));
}

// Approximate width of a label character, and the minimum gap between labels,
// as fractions of the colour bar width (0.8rem text on a 200 px bar)
const CHAR_WIDTH = 0.036;
const GAP = 0.04;

/** Extent of a label on the bar: it is shifted left by t of its width, so the end labels stay inside. */
function extent(tick: Tick): [number, number] {
  const width = tick.label.length * CHAR_WIDTH;
  const left = tick.t - tick.t * width;
  return [left, left + width];
}

function overlaps(a: Tick, b: Tick): boolean {
  const [a0, a1] = extent(a);
  const [b0, b1] = extent(b);
  return a0 < b1 + GAP && b0 < a1 + GAP;
}

/**
 * Ticks of a legend: both ends, the split of a split log scale, and round
 * values in between (powers of ten, then their multiples by 2 and 5 on a log
 * scale) where their labels don't overlap others.
 */
export function ticks(scale: Scale): Tick[] {
  const out: Tick[] = [
    { t: 0, label: format(scale.min) },
    { t: 1, label: format(scale.max) },
  ];
  // The split is always labelled, on a second line if it overlaps an end
  const split = scale.log ? scale.split : undefined;
  if (split !== undefined && split > scale.min && split < scale.max) {
    const tick: Tick = { t: scaleT(scale, split), label: format(split) };
    if (out.some((o) => overlaps(o, tick))) tick.row = 1;
    out.push(tick);
  }

  const inner: number[] = [];
  if (scale.log) {
    for (const factor of [1, 2, 5]) {
      for (let k = Math.floor(Math.log10(scale.min)); 10 ** k < scale.max; k++) {
        const v = factor * 10 ** k;
        if (v > scale.min && v < scale.max) inner.push(v);
      }
    }
  } else {
    const step = 10 ** Math.floor(Math.log10((scale.max - scale.min) / 2));
    const n = (scale.max - scale.min) / step > 5 ? 2 * step : step;
    for (let v = Math.ceil(scale.min / n) * n; v < scale.max; v += n) inner.push(v);
  }
  for (const v of inner) {
    const tick: Tick = { t: scaleT(scale, v), label: format(v) };
    if (out.every((o) => o.row || !overlaps(o, tick))) out.push(tick);
  }
  return out.sort((a, b) => a.t - b.t);
}
