/**
 * Stratigraphy of a line: a window with the profile of all its isochrones along the
 * distance, opened by clicking an isochrone or a track.
 *
 * In elevation (default): the ice surface, the isochrones (surface - depth) and the bed,
 * filled down to the bottom of the plot. In depth below the surface: the isochrone depths
 * and the ice thickness as the bed. The profile is exported per line by
 * anta_database.web.irh (tracks[].profile), and fetched when opened.
 *
 * Hovering a line gives its values, like hovering in 3D; anywhere in the plot, the position along
 * the line is marked in 3D. The wheel zooms along the distance, dragging pans, and a double
 * click shows all of it again.
 *
 * Profiles are at full resolution when small enough, else averaged along the distance
 * (profile.resolution, shown under Details). The plot draws about two points per pixel of
 * the visible distances, so long profiles stay fast; zooming in shows every point.
 */
import { fetchBytes } from "./terrain";

export interface ProfileInfo {
  file: string;
  columns: string[]; // x, y, distance, surface, bed, thickness; then one depth per layer
  layers: number[]; // feature ids
  bed: string; // "BED_ELEV" or "surface - thickness"
  thickness: string; // "ICE_THK" (completed with Bedmap3) or "Bedmap3"
  resolution?: {
    points: number;
    raw_points: number;
    bin: number | null; // m, null at full resolution
    raw_spacing: number | null; // m, median spacing of the points of the line
  };
}

export interface ProfileLine {
  dataset: string;
  line_id: string;
  citation: string | null;
  surface: string; // "SURF_ELEV" or "Bedmap3"
  profile?: ProfileInfo;
}

/** Values at a point of the profile. */
export interface ProfilePoint {
  x: number; // km
  y: number; // km
  distance: number; // km
  surface: number; // m, NaN where unknown
  bed: number;
  thickness: number;
}

export interface Tooltip {
  title: string;
  rows: [string, string][];
  note: string | null;
}

export interface ProfileOptions {
  /** CSS colour of a layer (feature id). */
  colorOf(feature: number): string;
  /** Tooltip of a layer at a point, with its depth (m). */
  describeLayer(feature: number, depth: number, point: ProfilePoint): Tooltip;
  showTooltip(tooltip: Tooltip, clientX: number, clientY: number): void;
  hideTooltip(): void;
  /** The hovered point, to mark it in 3D; null when nothing is hovered. */
  onPoint(point: ProfilePoint | null): void;
  /** The window was closed (button or Escape). */
  onClose(): void;
  /** Facts about the line, first in the Details: its isochrones and their ages (for touch screens, without hover). */
  describeLine(line: ProfileLine, layers: number[]): [string, string][];
}

type Mode = "Elevation" | "Depth";
type Series = { kind: "layer"; index: number } | { kind: "surface" } | { kind: "bed" };

const COLUMNS = 6; // x, y, distance, surface, bed, thickness
const MARGIN = { left: 60, right: 14, top: 12, bottom: 38 };
const BED_FILL = "#8a6a4b";
const BED_LINE = "#5b4430";
const ICE_FILL = "rgba(160, 200, 235, 0.22)";
const SURFACE_LINE = "#5d9fd6";
const HOVER_PX = 10;

/** Round values between min and max, about count of them. */
function niceTicks(min: number, max: number, count: number): number[] {
  const span = max - min;
  if (!(span > 0)) return [min];
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((f) => f * power).find((s) => s >= raw) ?? 10 * power;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function button(text: string, title: string): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = text;
  b.title = title;
  b.setAttribute("aria-label", title);
  return b;
}

export class ProfilePanel {
  private el = document.createElement("section");
  private titleEl = document.createElement("div");
  private noteEl = document.createElement("span");
  private detailsEl = document.createElement("details");
  private detailsText = document.createElement("div");
  private canvas = document.createElement("canvas");
  private modeButtons: Record<Mode, HTMLButtonElement>;
  private expandButton = button("⤢", "Enlarge");
  // Reduced to its title, the plot hidden
  private reduceButton = button("−", "Reduce");
  private mode: Mode = "Elevation";
  // The line shown, null when closed
  line: ProfileLine | null = null;
  private rows = new Float32Array(0);
  private width = COLUMNS; // values per row
  private layers: number[] = [];
  // Visible distances (km), and all of them
  private view: [number, number] = [0, 1];
  private full: [number, number] = [0, 1];
  // The row under the pointer, and the line hovered there (null between the lines)
  private hover: { series: Series | null; row: number } | null = null;
  private pointer: { x: number; y: number; clientX: number; clientY: number } | null = null;
  private drag: { x: number; view: [number, number] } | null = null;
  private frame = 0;
  private request = 0; // the latest open(), whose data is shown
  // Number of gap rows (NaN, between parts of the line) up to each row, and the range of
  // values of the current view
  private gapsBefore = new Uint32Array(0);
  private range: { key: string; lo: number; hi: number } | null = null;

  constructor(private options: ProfileOptions) {
    this.el.id = "profile";
    this.el.hidden = true;
    this.el.setAttribute("aria-label", "Stratigraphy of the line");
    const header = document.createElement("header");
    const heading = document.createElement("div");
    heading.className = "heading";
    this.titleEl.className = "title";
    const note = document.createElement("div");
    note.className = "note";
    this.noteEl.className = "summary";
    const summary = document.createElement("summary");
    summary.textContent = "Details";
    this.detailsText.className = "details-text";
    this.detailsEl.append(summary, this.detailsText);
    note.append(this.noteEl, this.detailsEl);
    heading.append(this.titleEl, note);
    const modes = document.createElement("div");
    modes.className = "modes";
    this.modeButtons = {
      Elevation: button("Elevation", "Elevation (m)"),
      Depth: button("Depth", "Depth below the ice surface (m)"),
    };
    for (const mode of ["Elevation", "Depth"] as Mode[]) {
      this.modeButtons[mode].addEventListener("click", () => this.setMode(mode));
      modes.append(this.modeButtons[mode]);
    }
    const closeButton = button("✕", "Close");
    closeButton.addEventListener("click", () => this.close());
    this.expandButton.addEventListener("click", () => this.setExpanded(!this.el.classList.contains("expanded")));
    const buttons = document.createElement("div");
    buttons.className = "buttons";
    this.reduceButton.addEventListener("click", () => this.setReduced(!this.el.classList.contains("reduced")));
    buttons.append(modes, this.expandButton, this.reduceButton, closeButton);
    header.append(heading, buttons);
    const plot = document.createElement("div");
    plot.className = "plot";
    plot.append(this.canvas);
    this.el.append(header, plot);
    document.body.append(this.el);
    this.setMode("Elevation");

    new ResizeObserver(() => this.redraw()).observe(plot);
    this.canvas.addEventListener("pointermove", (e) => this.onPointerMove(e));
    this.canvas.addEventListener("pointerleave", (e) => {
      // A finger lifted: the tooltip of its tap stays, until the next touch
      if (e.pointerType === "touch") return;
      this.pointer = null;
      this.drag = null;
      this.setHover(null);
    });
    this.canvas.addEventListener("pointerdown", (e) => {
      this.canvas.setPointerCapture(e.pointerId);
      this.drag = { x: e.offsetX, view: [...this.view] };
      if (e.pointerType === "touch") this.setHover(null);
    });
    this.canvas.addEventListener("pointerup", (e) => {
      const tap = e.pointerType === "touch" && this.drag !== null && Math.abs(e.offsetX - this.drag.x) < 8;
      this.drag = null;
      // No hover on a touch screen: a tap shows what is under the finger
      if (tap) this.onPointerMove(e);
    });
    this.canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    this.canvas.addEventListener("dblclick", () => {
      this.view = [...this.full];
      this.redraw();
    });
    window.addEventListener("keydown", (e) => {
      if (e.key !== "Escape" || this.el.hidden) return;
      if (this.el.classList.contains("expanded")) this.setExpanded(false);
      else this.close();
    });
  }

  get isOpen(): boolean {
    return !this.el.hidden;
  }

  /** Show the profile of a line, fetched from baseUrl. */
  async open(line: ProfileLine, baseUrl: string): Promise<void> {
    if (!line.profile) throw new Error("No profile in this export: run anta_database.web.irh again");
    const request = ++this.request;
    this.line = line;
    this.titleEl.textContent = `${line.citation ?? line.dataset}, line ${line.line_id}`;
    this.noteEl.textContent = "Loading…";
    this.detailsEl.hidden = true;
    this.detailsEl.open = false;
    this.rows = new Float32Array(0);
    this.el.hidden = false;
    this.redraw();
    const rows = new Float32Array(await fetchBytes(`${baseUrl}/${line.profile.file}`));
    if (request !== this.request) return; // another line was opened meanwhile
    this.rows = rows;
    this.layers = line.profile.layers;
    this.width = COLUMNS + this.layers.length;
    const n = rows.length / this.width;
    this.full = [rows[2], rows[(n - 1) * this.width + 2]];
    this.view = [...this.full];
    this.range = null;
    // Gap rows: NaN everywhere but the distance
    this.gapsBefore = new Uint32Array(n);
    for (let r = 0, gaps = 0; r < n; r++) {
      if (Number.isNaN(rows[r * this.width])) gaps++;
      this.gapsBefore[r] = gaps;
    }
    this.describe(line, line.profile, n - this.gapsBefore[n - 1]);
    this.redraw();
  }

  /** The short description, and what the Details tell. */
  private describe(line: ProfileLine, profile: ProfileInfo, points: number): void {
    const count = (n: number) => n.toLocaleString("en");
    const metres = (m: number) => `${m < 10 ? m.toFixed(1) : Math.round(m).toLocaleString("en")} m`;
    const r = profile.resolution;
    const spacing = r?.raw_spacing ? `, every ${metres(r.raw_spacing)} (median)` : "";
    let resolution: string;
    if (!r) {
      this.noteEl.textContent = `${count(points)} points`;
      resolution = `${count(points)} points along the line.`;
    } else if (r.bin === null) {
      this.noteEl.textContent = `Full resolution, ${count(points)} points`;
      resolution = `Full resolution: all ${count(r.raw_points)} points of the line${spacing}.`;
    } else {
      this.noteEl.textContent = `Averaged in ${metres(r.bin)} bins, ${count(points)} points`;
      resolution =
        `Averaged in bins of ${metres(r.bin)} along the line, the smallest that keep the profile under ` +
        `about 4 MB: ${count(points)} points from ${count(r.raw_points)}${spacing}. Each value is the mean of ` +
        "the points of its bin.";
    }
    const surface = line.surface === "SURF_ELEV" ? "SURF_ELEV of the dataset" : `${line.surface} surface elevation (500 m grid)`;
    const thickness = profile.thickness === "ICE_THK" ? "ICE_THK of the dataset, completed with Bedmap3 where missing" : `${profile.thickness} ice thickness (500 m grid)`;
    const bed = profile.bed === "BED_ELEV" ? "BED_ELEV of the dataset" : "surface elevation minus ice thickness";
    const items: [string, string][] = [
      ...this.options.describeLine(line, profile.layers),
      ["Resolution", resolution],
      ["Surface", surface],
      ["Ice thickness", thickness],
      ["Bed", bed],
      ["Isochrones", "elevation = surface - IRH_DEPTH; depth = IRH_DEPTH"],
      ["Plot", "scroll to zoom along the line, drag to pan, double-click to see it all; on a touch screen, tap a line for its details. Long profiles are drawn with about two points per pixel: zoom in to see every point."],
    ];
    this.detailsText.replaceChildren(
      ...items.map(([key, text]) => {
        const row = document.createElement("div");
        const k = document.createElement("span");
        k.className = "key";
        k.textContent = `${key}: `;
        row.append(k, text);
        return row;
      }),
    );
    this.detailsEl.hidden = false;
  }

  close(): void {
    if (this.el.hidden) return;
    this.el.hidden = true;
    this.setExpanded(false);
    this.setReduced(false);
    this.line = null;
    this.request++;
    this.setHover(null);
    this.options.onClose();
  }

  private setReduced(reduced: boolean): void {
    if (reduced) this.setExpanded(false);
    this.el.classList.toggle("reduced", reduced);
    this.reduceButton.textContent = reduced ? "+" : "−";
    const label = reduced ? "Show the plot" : "Reduce";
    this.reduceButton.title = label;
    this.reduceButton.setAttribute("aria-label", label);
    if (reduced) this.setHover(null);
    else this.redraw();
  }

  private setMode(mode: Mode): void {
    this.mode = mode;
    for (const [m, b] of Object.entries(this.modeButtons)) b.classList.toggle("active", m === mode);
    this.hover = null;
    this.redraw();
  }

  /** Points of the line shown (x, y in km, NaN between its parts), empty while loading. */
  points(): { x: Float32Array; y: Float32Array } {
    const x = new Float32Array(this.count);
    const y = new Float32Array(this.count);
    for (let row = 0; row < this.count; row++) {
      x[row] = this.value(row, 0);
      y[row] = this.value(row, 1);
    }
    return { x, y };
  }

  private setExpanded(expanded: boolean): void {
    if (expanded) this.setReduced(false);
    this.el.classList.toggle("expanded", expanded);
    this.expandButton.textContent = expanded ? "⤡" : "⤢";
    const label = expanded ? "Make smaller" : "Enlarge";
    this.expandButton.title = label;
    this.expandButton.setAttribute("aria-label", label);
  }

  private redraw(): void {
    cancelAnimationFrame(this.frame);
    this.frame = requestAnimationFrame(() => this.draw());
  }

  private get count(): number {
    return this.rows.length / this.width;
  }

  private value(row: number, column: number): number {
    return this.rows[row * this.width + column];
  }

  /** Value of a series at a row, in the units of the mode (NaN where unknown). */
  private seriesValue(series: Series, row: number): number {
    const surface = this.value(row, 3);
    if (series.kind === "surface") return this.mode === "Elevation" ? surface : Number.isFinite(surface) ? 0 : NaN;
    if (series.kind === "bed") return this.mode === "Elevation" ? this.value(row, 4) : this.value(row, 5);
    const depth = this.value(row, COLUMNS + series.index);
    return this.mode === "Elevation" ? surface - depth : depth;
  }

  private allSeries(): Series[] {
    return [...this.layers.map((_, index): Series => ({ kind: "layer", index })), { kind: "surface" }, { kind: "bed" }];
  }

  /** First row at a distance (km) or after it. */
  private rowAt(distance: number): number {
    let lo = 0;
    let hi = this.count;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.value(mid, 2) < distance) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /** Rows shown: first and last in the visible distances, and one more on each side. */
  private visibleRows(): [number, number] {
    const first = this.rowAt(this.view[0]);
    const last = this.rowAt(this.view[1]) - 1;
    return [Math.max(0, first - 1), Math.min(this.count - 1, Math.max(first, last + 1))];
  }

  /** Plot area and scales of the current view. */
  private frameOf(width: number, height: number) {
    const [first, last] = this.visibleRows();
    // The range of values only changes with the view and the mode
    const key = `${this.mode} ${first} ${last}`;
    if (this.range?.key !== key) {
      let min = Infinity;
      let max = -Infinity;
      for (let r = first; r <= last; r++) {
        for (const s of this.allSeries()) {
          const v = this.seriesValue(s, r);
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
      this.range = { key, lo: min, hi: max };
    }
    let { lo, hi } = this.range;
    if (!Number.isFinite(lo)) [lo, hi] = [0, 1];
    if (this.mode === "Depth") lo = 0;
    const pad = 0.05 * (hi - lo || 100);
    if (this.mode === "Elevation") lo -= pad;
    hi += pad;
    const area = {
      left: MARGIN.left,
      top: MARGIN.top,
      width: Math.max(10, width - MARGIN.left - MARGIN.right),
      height: Math.max(10, height - MARGIN.top - MARGIN.bottom),
    };
    const [d0, d1] = this.view;
    const px = (d: number) => area.left + ((d - d0) / (d1 - d0 || 1)) * area.width;
    // Elevation upwards, depth downwards
    const py = (v: number) =>
      area.top + (this.mode === "Elevation" ? (hi - v) / (hi - lo) : (v - lo) / (hi - lo)) * area.height;
    return { area, lo, hi, px, py, first, last };
  }

  private draw(): void {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height || this.el.hidden) return;
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    const ctx = this.canvas.getContext("2d")!;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const style = getComputedStyle(this.el);
    const text = style.getPropertyValue("--text").trim() || "#222";
    const muted = style.getPropertyValue("--muted").trim() || "#777";
    ctx.font = "12px system-ui, sans-serif";
    if (!this.count) {
      ctx.fillStyle = muted;
      ctx.fillText(this.line ? "Loading the profile…" : "", MARGIN.left, height / 2);
      return;
    }
    const { area, lo, hi, px, py, first, last } = this.frameOf(width, height);

    // Grid and axes
    ctx.save();
    ctx.strokeStyle = muted;
    ctx.globalAlpha = 0.25;
    ctx.lineWidth = 1;
    const yTicks = niceTicks(Math.min(lo, hi), Math.max(lo, hi), Math.max(3, Math.round(area.height / 45)));
    const xTicks = niceTicks(this.view[0], this.view[1], Math.max(3, Math.round(area.width / 90)));
    ctx.beginPath();
    for (const v of yTicks) {
      ctx.moveTo(area.left, Math.round(py(v)) + 0.5);
      ctx.lineTo(area.left + area.width, Math.round(py(v)) + 0.5);
    }
    for (const d of xTicks) {
      ctx.moveTo(Math.round(px(d)) + 0.5, area.top);
      ctx.lineTo(Math.round(px(d)) + 0.5, area.top + area.height);
    }
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.beginPath();
    ctx.rect(area.left, area.top, area.width, area.height);
    ctx.clip();
    const bottom = area.top + area.height;
    // Rows drawn: about two per pixel; lines never join across a gap of the line
    const stride = Math.max(1, Math.floor((last - first) / (2 * area.width)));
    const samples: number[] = [];
    for (let r = first; r < last; r += stride) samples.push(r);
    samples.push(last);
    // Runs (first and last sample) where a series is known
    const runs = (series: Series): [number, number][] => {
      const out: [number, number][] = [];
      let start = -1;
      for (let k = 0; k <= samples.length; k++) {
        const r = samples[k];
        const known =
          k < samples.length &&
          Number.isFinite(this.seriesValue(series, r)) &&
          (start < 0 || this.gapsBefore[r] === this.gapsBefore[samples[k - 1]]);
        if (known && start < 0) start = k;
        else if (!known && start >= 0) {
          out.push([start, k - 1]);
          start = -1;
          if (k < samples.length && Number.isFinite(this.seriesValue(series, r))) start = k; // after a gap
        }
      }
      return out;
    };
    const path = (series: Series, from: number, to: number) => {
      for (let k = from; k <= to; k++) {
        const r = samples[k];
        const [x, y] = [px(this.value(r, 2)), py(this.seriesValue(series, r))];
        if (k === from) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
    };
    const sampleDistance = (k: number) => this.value(samples[k], 2);
    // Ice: from the surface down to the bed, where both are known
    const surface: Series = { kind: "surface" };
    const bed: Series = { kind: "bed" };
    ctx.fillStyle = ICE_FILL;
    for (const [a, b] of runs(surface)) {
      ctx.beginPath();
      path(surface, a, b);
      for (let k = b; k >= a; k--) {
        const v = this.seriesValue(bed, samples[k]);
        ctx.lineTo(px(sampleDistance(k)), Number.isFinite(v) ? py(v) : bottom);
      }
      ctx.closePath();
      ctx.fill();
    }
    // Bed, filled down to the bottom of the plot
    for (const [a, b] of runs(bed)) {
      ctx.beginPath();
      path(bed, a, b);
      ctx.lineTo(px(sampleDistance(b)), bottom);
      ctx.lineTo(px(sampleDistance(a)), bottom);
      ctx.closePath();
      ctx.fillStyle = BED_FILL;
      ctx.fill();
      ctx.beginPath();
      path(bed, a, b);
      ctx.strokeStyle = BED_LINE;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    const stroke = (series: Series, color: string, width: number) => {
      ctx.beginPath();
      for (const [a, b] of runs(series)) path(series, a, b);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
    };
    stroke(surface, SURFACE_LINE, 1.5);
    const hovered = this.hover?.series;
    this.layers.forEach((feature, index) => {
      const isHovered = hovered?.kind === "layer" && hovered.index === index;
      if (!isHovered) stroke({ kind: "layer", index }, this.options.colorOf(feature), 1.6);
    });
    if (hovered) {
      const color = hovered.kind === "layer" ? this.options.colorOf(this.layers[hovered.index]) : hovered.kind === "bed" ? BED_LINE : SURFACE_LINE;
      stroke(hovered, "#ff4d4d", 4);
      stroke(hovered, color, 2);
    }
    // Cursor: a vertical line, and the hovered point
    if (this.hover) {
      const x = px(this.value(this.hover.row, 2));
      ctx.strokeStyle = muted;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, area.top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      if (this.hover.series) {
        const y = py(this.seriesValue(this.hover.series, this.hover.row));
        ctx.fillStyle = "#ff4d4d";
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    ctx.restore();

    // Labels
    ctx.fillStyle = text;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of yTicks) ctx.fillText(v.toLocaleString("en"), area.left - 6, py(v));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const d of xTicks) ctx.fillText(d.toLocaleString("en"), px(d), bottom + 5);
    ctx.fillStyle = muted;
    ctx.fillText("Distance along the line (km)", area.left + area.width / 2, bottom + 21);
    ctx.save();
    ctx.translate(13, area.top + area.height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(this.mode === "Elevation" ? "Elevation (m)" : "Depth below the surface (m)", 0, 0);
    ctx.restore();
  }

  private onWheel(event: WheelEvent): void {
    if (!this.count) return;
    event.preventDefault();
    const { area } = this.frameOf(this.canvas.clientWidth, this.canvas.clientHeight);
    const [d0, d1] = this.view;
    const at = d0 + ((event.offsetX - area.left) / area.width) * (d1 - d0);
    const factor = Math.exp(event.deltaY * 0.0015);
    const span = Math.min(this.full[1] - this.full[0], Math.max(0.2, (d1 - d0) * factor));
    this.setView(at - ((at - d0) / (d1 - d0)) * span, span);
  }

  /** Show span km of distance from start, inside the line. */
  private setView(start: number, span: number): void {
    const [f0, f1] = this.full;
    start = Math.min(Math.max(start, f0), f1 - span);
    this.view = [start, start + span];
    this.redraw();
  }

  private onPointerMove(event: PointerEvent): void {
    this.pointer = { x: event.offsetX, y: event.offsetY, clientX: event.clientX, clientY: event.clientY };
    if (!this.count) return;
    const { area, px, py } = this.frameOf(this.canvas.clientWidth, this.canvas.clientHeight);
    if (this.drag && event.buttons) {
      const span = this.drag.view[1] - this.drag.view[0];
      this.setView(this.drag.view[0] - ((event.offsetX - this.drag.x) / area.width) * span, span);
      this.setHover(null);
      return;
    }
    // Row closest to the cursor along the distance
    const d = this.view[0] + ((event.offsetX - area.left) / area.width) * (this.view[1] - this.view[0]);
    const hi = Math.min(this.count - 1, this.rowAt(d));
    const lo = Math.max(0, hi - 1);
    const row = Math.abs(this.value(lo, 2) - d) <= Math.abs(this.value(hi, 2) - d) ? lo : hi;
    // Series closest to the cursor at that row, layers first when as close
    let best: { series: Series; distance: number } | null = null;
    if (Math.abs(px(this.value(row, 2)) - event.offsetX) < 3 * HOVER_PX) {
      for (const series of this.allSeries()) {
        const v = this.seriesValue(series, row);
        if (!Number.isFinite(v)) continue;
        const distance = Math.abs(py(v) - event.offsetY);
        if (distance <= HOVER_PX && (!best || distance < best.distance - 0.5)) best = { series, distance };
      }
    }
    // Between the lines, still the position along the line (marked in 3D), without a tooltip
    const inside = event.offsetX >= area.left && event.offsetX <= area.left + area.width;
    this.setHover(best ? { series: best.series, row } : inside ? { series: null, row } : null);
  }

  private setHover(hover: { series: Series | null; row: number } | null): void {
    const same = hover && this.hover && hover.row === this.hover.row && JSON.stringify(hover.series) === JSON.stringify(this.hover.series);
    if (!same) {
      this.hover = hover;
      this.redraw();
    }
    if (!hover || !this.pointer) {
      this.options.hideTooltip();
      this.options.onPoint(null);
      return;
    }
    const r = hover.row;
    const point: ProfilePoint = {
      x: this.value(r, 0),
      y: this.value(r, 1),
      distance: this.value(r, 2),
      surface: this.value(r, 3),
      bed: this.value(r, 4),
      thickness: this.value(r, 5),
    };
    const at = ["Distance", `${point.distance.toFixed(2)} km`] as [string, string];
    const round = (v: number) => (Number.isFinite(v) ? `${Math.round(v)} m` : "unknown");
    this.options.onPoint(point);
    if (!hover.series) {
      this.options.hideTooltip();
      return;
    }
    let tooltip: Tooltip;
    if (hover.series.kind === "layer") {
      const depth = this.value(r, COLUMNS + hover.series.index);
      tooltip = this.options.describeLayer(this.layers[hover.series.index], depth, point);
      tooltip.rows.push(at);
    } else if (hover.series.kind === "bed") {
      tooltip = {
        title: "Bed",
        rows: [["Bed elevation", round(point.bed)], ["Ice thickness", round(point.thickness)], at],
        note: null,
      };
    } else {
      tooltip = { title: "Ice surface", rows: [["Surface elevation", round(point.surface)], at], note: null };
    }
    this.options.showTooltip(tooltip, this.pointer.clientX, this.pointer.clientY);
  }
}
