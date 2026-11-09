/**
 * A small counter of the hidden things found (eggs.ts): clicking one collects it, with a short
 * animation, and it stays in place. What was found is kept in the browser (local storage), apart
 * from the settings of the viewer, so resetting them does not lose it. The total comes from the
 * eggs of the viewer, so new ones count at once for those who found the others. Clicking the
 * counter opens a small window with the eggs found, and a button to start again.
 */
const KEY = "eggs-found";
const FLIGHT_MS = 800;
const NAME_MS = 2800;

/** How to play: in the tutorial, and in the window of the counter until an egg is found. */
export const EGGS_HINT =
  "A few strange things are hidden somewhere in Antarctica: on the ice, on the bed and in the ocean. Zoom in close: they are " +
  "tiny from afar. **Click** one to collect it: this small counter keeps track of those you found.";

export class EggCounter {
  readonly el = document.createElement("button");
  private panel = document.createElement("section");
  private count = document.createElement("span");
  private name = document.createElement("span");
  private found: Set<string>;
  private nameTimer = 0;

  constructor(private eggs: { id: string; title: string }[]) {
    this.found = new Set(this.load());
    this.el.id = "egg-counter";
    this.el.type = "button";
    this.el.setAttribute("aria-haspopup", "dialog");
    this.el.addEventListener("click", () => this.toggle(this.panel.hidden !== false));
    this.count.setAttribute("role", "status");
    const icon = document.createElement("span");
    icon.className = "egg-icon";
    this.name.className = "egg-name";
    this.el.append(this.name, icon, this.count);
    this.panel.className = "tutorial egg-panel";
    this.panel.setAttribute("role", "dialog");
    this.panel.setAttribute("aria-label", "Eggs found");
    this.panel.hidden = true;
    document.body.append(this.el, this.panel);
    // Before the keys that move the view, as the tutorial does
    window.addEventListener(
      "keydown",
      (e) => {
        if (e.key !== "Escape" || this.panel.hidden) return;
        this.toggle(false);
        e.stopPropagation();
      },
      true,
    );
    this.show();
  }

  private load(): string[] {
    try {
      const saved = JSON.parse(localStorage.getItem(KEY) ?? "[]");
      return Array.isArray(saved) ? saved.filter((id) => typeof id === "string") : [];
    } catch {
      return []; // storage blocked, or unreadable
    }
  }

  private show(): void {
    const found = this.eggs.filter((egg) => this.found.has(egg.id)).length;
    const total = this.eggs.length;
    this.count.textContent = `${found}/${total}`;
    this.el.classList.toggle("complete", found === total);
    this.el.title =
      found === total
        ? "You found all the eggs hidden in Antarctica! Click to see them."
        : `${found} of ${total} eggs found. Click for more.`;
    this.el.setAttribute("aria-label", `Eggs found: ${found} of ${total}`);
    if (!this.panel.hidden) this.fillPanel();
  }

  private toggle(open: boolean): void {
    this.panel.hidden = !open;
    if (open) this.fillPanel();
  }

  private fillPanel(): void {
    const button = (className: string, text: string, title: string, onClick: () => void) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = className;
      b.textContent = text;
      b.title = title;
      b.addEventListener("click", onClick);
      return b;
    };
    const heading = document.createElement("h2");
    const close = button("close", "×", "Close", () => this.toggle(false));
    const eggs = this.eggs.filter((egg) => this.found.has(egg.id));
    const left = this.eggs.length - eggs.length;
    heading.textContent = `Eggs found: ${eggs.length}/${this.eggs.length}`;
    const text = document.createElement("p");
    if (eggs.length === 0) {
      // Bold between ** **, the rest as text
      EGGS_HINT.split(/\*\*(.+?)\*\*/).forEach((part, i) => {
        const strong = document.createElement("strong");
        strong.textContent = part;
        text.append(i % 2 ? strong : part);
      });
      this.panel.replaceChildren(close, heading, text);
      return;
    }
    const list = document.createElement("ul");
    list.append(
      ...eggs.map((egg) => {
        const item = document.createElement("li");
        item.textContent = egg.title;
        return item;
      }),
    );
    text.textContent = left === 0 ? "You found them all!" : `${left} more to find.`;
    const buttons = document.createElement("div");
    buttons.className = "buttons";
    buttons.append(button("", "Reset counter", "Forget the eggs found, and start again", () => this.reset()));
    this.panel.replaceChildren(close, heading, list, text, buttons);
  }

  /** Forget the eggs found: they can be collected again. */
  private reset(): void {
    this.found.clear();
    try {
      localStorage.removeItem(KEY);
    } catch {
      // storage blocked: nothing was kept
    }
    this.show();
  }

  has(id: string): boolean {
    return this.found.has(id);
  }

  /** Collect an egg clicked at a point of the window (px). */
  collect(id: string, title: string, x: number, y: number): void {
    this.found.add(id);
    try {
      localStorage.setItem(KEY, JSON.stringify([...this.found]));
    } catch {
      // storage blocked: found until the page is left
    }
    const done = () => {
      this.show();
      this.el.animate([{ transform: "scale(1)" }, { transform: "scale(1.3)" }, { transform: "scale(1)" }], { duration: 350 });
      this.name.textContent = title;
      this.el.classList.add("named");
      clearTimeout(this.nameTimer);
      this.nameTimer = window.setTimeout(() => this.el.classList.remove("named"), NAME_MS);
    };
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      done();
      return;
    }
    this.burst(x, y);
    // A small egg flies from the click to the counter
    const egg = document.createElement("span");
    egg.className = "egg-icon egg-flying";
    egg.style.left = `${x}px`;
    egg.style.top = `${y}px`;
    document.body.append(egg);
    const to = this.count.getBoundingClientRect();
    const [dx, dy] = [to.left - x, to.top + to.height / 2 - y];
    egg.animate(
      [
        { transform: "translate(-50%, -50%) scale(0.6)", opacity: 1 },
        { transform: "translate(-50%, -50%) scale(2.2)", opacity: 1, offset: 0.3 },
        { transform: `translate(calc(${dx}px - 50%), calc(${dy}px - 50%)) scale(1)`, opacity: 0.9 },
      ],
      { duration: FLIGHT_MS, easing: "ease-in" },
    );
    // By a timer: an animation does not end in a tab left in the background
    window.setTimeout(() => {
      egg.remove();
      done();
    }, FLIGHT_MS);
  }

  /** A ring and a few sparks around the place of the click. */
  private burst(x: number, y: number): void {
    const SPARKS = 8;
    for (let k = 0; k <= SPARKS; k++) {
      const part = document.createElement("span");
      part.className = k === SPARKS ? "egg-ring" : "egg-spark";
      part.style.left = `${x}px`;
      part.style.top = `${y}px`;
      document.body.append(part);
      const a = (k / SPARKS) * 2 * Math.PI;
      const end =
        k === SPARKS
          ? "translate(-50%, -50%) scale(1)"
          : `translate(calc(${34 * Math.cos(a)}px - 50%), calc(${34 * Math.sin(a)}px - 50%)) scale(0.3)`;
      part.animate(
        [
          { transform: k === SPARKS ? "translate(-50%, -50%) scale(0.1)" : "translate(-50%, -50%)", opacity: 1 },
          { transform: end, opacity: 0 },
        ],
        { duration: 600, easing: "ease-out" },
      );
      window.setTimeout(() => part.remove(), 600);
    }
  }
}
