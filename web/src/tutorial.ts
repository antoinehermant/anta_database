/**
 * Tour of the viewer: a few bubbles pointing at what they explain, ending with the hidden
 * things to find. It starts by itself on the first visit; once finished or closed, the browser
 * remembers it (localStorage, nothing is sent anywhere) and it only starts again from its
 * button in the menu.
 */

const SEEN_KEY = "tutorial";

export interface TutorialStep {
  title: string;
  text: string;
  /** Something to try, under the text. */
  tip?: string;
  /** What the step points at; the bubble is in the middle of the page without it. */
  target?: () => Element | null | undefined;
  /** Before the step: open what it points at. */
  prepare?: () => void;
}

/** Whether the tutorial was already finished or closed in this browser. */
export function tutorialSeen(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === "seen";
  } catch {
    return true; // storage blocked: never start by itself, it would come back on every visit
  }
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = className;
  e.textContent = text;
  return e;
}

export class Tutorial {
  private bubble = element("section", "tutorial");
  private ring = element("div", "tutorial-ring");
  private index = 0;
  private open = false;
  // What the current step points at, marked with the class tutorial-target
  private target: Element | null = null;

  constructor(private steps: TutorialStep[]) {
    this.bubble.setAttribute("role", "dialog");
    this.bubble.setAttribute("aria-label", "Tutorial");
    this.bubble.hidden = this.ring.hidden = true;
    document.body.append(this.ring, this.bubble);
    window.addEventListener("resize", () => this.place());
    // The menu scrolled under the ring
    document.addEventListener("scroll", () => this.place(), true);
    // Before the keys that move the view: they keep working, but not Escape and Enter
    window.addEventListener(
      "keydown",
      (e) => {
        if (!this.open) return;
        if (e.key === "Escape") this.close();
        else if (e.key === "Enter" && !(e.target as HTMLElement).closest("button, input, select")) this.go(this.index + 1);
        else return;
        e.preventDefault();
        e.stopPropagation();
      },
      true,
    );
  }

  start(): void {
    this.open = true;
    this.go(0);
  }

  /** Closed or finished: not shown again by itself. */
  close(): void {
    this.open = false;
    this.bubble.hidden = this.ring.hidden = true;
    this.mark(null);
    try {
      localStorage.setItem(SEEN_KEY, "seen");
    } catch {
      // storage blocked: nothing to remember it with
    }
  }

  private go(index: number): void {
    if (index >= this.steps.length) {
      this.close();
      return;
    }
    this.index = Math.max(0, index);
    const step = this.steps[this.index];
    step.prepare?.();
    const last = this.index === this.steps.length - 1;
    const close = element("button", "close", "×");
    close.type = "button";
    close.title = "Close the tutorial";
    close.addEventListener("click", () => this.close());
    const buttons = element("div", "buttons");
    buttons.append(element("span", "count", `${this.index + 1} / ${this.steps.length}`));
    if (this.index > 0) {
      const back = element("button", "", "Back");
      back.type = "button";
      back.addEventListener("click", () => this.go(this.index - 1));
      buttons.append(back);
    }
    const next = element("button", "next", last ? "Let's go!" : this.index === 0 ? "Show me" : "Next");
    next.type = "button";
    next.addEventListener("click", () => this.go(this.index + 1));
    buttons.append(next);
    const text = element("p");
    // Bold between ** **, the rest as text
    step.text.split(/\*\*(.+?)\*\*/).forEach((part, i) => text.append(i % 2 ? element("strong", "", part) : part));
    const tip = step.tip ? [element("p", "tip", step.tip)] : [];
    if (step.tip) tip[0].prepend(element("strong", "", "Tip: "));
    this.bubble.replaceChildren(close, element("h2", "", step.title), text, ...tip, buttons);
    this.bubble.hidden = false;
    // After what prepare() opened is laid out
    requestAnimationFrame(() => this.place());
    next.focus({ preventScroll: true });
  }

  private mark(target: Element | null): void {
    this.target?.classList.remove("tutorial-target");
    target?.classList.add("tutorial-target");
    this.target = target;
  }

  /** Put the bubble next to the target of the step, and the ring around it. */
  private place(): void {
    if (!this.open) return;
    const target = this.steps[this.index].target?.();
    const rect = target instanceof HTMLElement && target.getClientRects().length ? target.getBoundingClientRect() : null;
    this.mark(rect ? target! : null);
    const b = this.bubble.getBoundingClientRect();
    const margin = 12;
    const [w, h] = [window.innerWidth, window.innerHeight];
    this.ring.hidden = !rect;
    this.bubble.classList.toggle("centred", !rect);
    if (!rect) {
      Object.assign(this.bubble.style, { left: `${(w - b.width) / 2}px`, top: `${(h - b.height) / 2}px` });
      return;
    }
    Object.assign(this.ring.style, {
      left: `${rect.left - 4}px`,
      top: `${rect.top - 4}px`,
      width: `${rect.width + 8}px`,
      height: `${rect.height + 8}px`,
    });
    // Right of the target, else left of it, else above or below, kept in the page
    let left = rect.right + margin;
    let top = rect.top;
    if (left + b.width > w - margin) left = rect.left - margin - b.width;
    if (left < margin) {
      left = Math.min(Math.max(margin, rect.left), w - margin - b.width);
      top = rect.top - margin - b.height >= margin ? rect.top - margin - b.height : rect.bottom + margin;
    }
    top = Math.min(Math.max(margin, top), h - margin - b.height);
    Object.assign(this.bubble.style, { left: `${Math.max(margin, left)}px`, top: `${top}px` });
  }
}
