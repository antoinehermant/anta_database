/** Colour bar legends, stacked in the #legends corner. */
import type { Colormap, Tick } from "./colormaps";

export class Legend {
  private el = document.createElement("div");

  constructor(container: HTMLElement) {
    this.el.className = "legend";
    this.el.hidden = true;
    container.append(this.el);
  }

  /** A colour bar, and below it swatches of single colours (label, CSS colour). */
  show(title: string, colormap: Colormap, ticks: Tick[], swatches: [string, string][] = []): void {
    const heading = document.createElement("div");
    heading.textContent = title;
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.background = colormap.css();
    const labels = document.createElement("div");
    labels.className = "ticks";
    labels.style.height = `${1.2 * (1 + Math.max(0, ...ticks.map((t) => t.row ?? 0)))}em`;
    for (const tick of ticks) {
      const label = document.createElement("span");
      label.textContent = tick.label;
      // Labels at the ends stay inside the bar
      label.style.left = `${tick.t * 100}%`;
      label.style.transform = `translateX(${-tick.t * 100}%)`;
      if (tick.row) label.style.top = `${tick.row * 1.2}em`;
      labels.append(label);
    }
    const items = swatches.map(([text, color]) => {
      const item = document.createElement("div");
      item.className = "swatch";
      const swatch = document.createElement("span");
      swatch.style.background = color;
      item.append(swatch, text);
      return item;
    });
    this.el.replaceChildren(heading, bar, labels, ...items);
    this.el.hidden = false;
  }

  hide(): void {
    this.el.hidden = true;
  }
}
