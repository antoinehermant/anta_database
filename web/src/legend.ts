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

  /**
   * Samples of lines: groups of (label, CSS colour, dash and gap in px or null for solid),
   * each group as a column.
   */
  showLines(title: string, groups: [label: string, color: string, dash: [number, number] | null][][]): void {
    const heading = document.createElement("div");
    heading.textContent = title;
    const columns = document.createElement("div");
    columns.className = "line-samples";
    for (const group of groups) {
      const column = document.createElement("div");
      for (const [text, color, dash] of group) {
        const item = document.createElement("div");
        item.className = "swatch";
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", "24");
        svg.setAttribute("height", "6");
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        Object.entries({ x1: "1", y1: "3", x2: "23", y2: "3", stroke: color, "stroke-width": "2.5", "stroke-linecap": "round" }).forEach(
          ([k, v]) => line.setAttribute(k, v),
        );
        if (dash) line.setAttribute("stroke-dasharray", `${dash[0]} ${dash[1] + 2.5}`);
        svg.append(line);
        item.append(svg, text);
        column.append(item);
      }
      columns.append(column);
    }
    this.el.replaceChildren(heading, columns);
    this.el.hidden = false;
  }

  hide(): void {
    this.el.hidden = true;
  }
}
