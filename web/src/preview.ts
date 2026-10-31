/**
 * Private 3D preview of a submission (viewer/?preview=<id>&key=<key>): the dataset sent through
 * the submission portal, compiled by its worker (portal/pipeline.py) and served by the portal
 * (portal/app.py, /api/previews) to whoever has the key, instead of the isochrones of the
 * database. While it is being compiled, a panel shows the progress; once ready, what the
 * automatic checks found.
 */

const PORTAL_URL = (import.meta.env.VITE_PORTAL_URL ?? "").replace(/\/$/, "");
const POLL_MS = 3000;

export interface PreviewStatus {
  status: "received" | "processing" | "ready" | "failed" | "closed";
  dataset: string;
  citation: string;
  licence: "redistribute" | "review";
  steps: string[];
  step: string | null;
  waiting?: number;
  warnings?: string[];
  errors?: string[];
  /** Files of the compiled dataset to download (path: size in bytes), once ready */
  downloads?: Record<string, number>;
  /** Extent of the dataset, EPSG:3031 (m): xmin, xmax, ymin, ymax */
  bbox?: [number, number, number, number] | null;
}

// Steps of portal/pipeline.py
const STEP_TEXT: Record<string, string> = {
  checking: "Checking the files",
  compiling: "Compiling the dataset",
  indexing: "Assigning the ages",
  reviewing: "Checking the compiled dataset",
  exporting: "Building the 3D view",
  storing: "Storing the compiled dataset",
};

const params = new URLSearchParams(location.search);
const id = params.get("preview");
const key = params.get("key");

/** The preview of the page, if any: where its files are. */
export const SUBMISSION =
  id && key ? { id, url: `${PORTAL_URL}/api/previews/${encodeURIComponent(id)}/${encodeURIComponent(key)}` } : null;

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text = ""): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = className;
  e.textContent = text;
  return e;
}

/** Issues of the checks, folded. */
function issues(title: string, list: string[], open = false): HTMLElement {
  const details = element("details", "issues");
  details.open = open;
  details.append(element("summary", "", title));
  const ul = element("ul");
  for (const text of list) ul.append(element("li", "", text));
  details.append(ul);
  return details;
}

/**
 * Reduce or expand a panel of the preview to its title, with a button in its corner and by
 * clicking the title: the panels are never closed, so they can always be found again. The state
 * is a class of the panel, kept when its content is replaced.
 */
export function reduceButton(panel: HTMLElement, title: HTMLElement): HTMLButtonElement {
  const button = element("button", "reduce");
  button.type = "button";
  const update = () => {
    const reduced = panel.classList.contains("reduced");
    button.textContent = reduced ? "+" : "−";
    button.title = reduced ? "Expand" : "Reduce";
    button.setAttribute("aria-expanded", String(!reduced));
  };
  const toggle = () => {
    panel.classList.toggle("reduced");
    update();
  };
  button.addEventListener("click", toggle);
  title.addEventListener("click", toggle);
  update();
  return button;
}

function show(panel: HTMLElement, s: PreviewStatus | null, problem = ""): void {
  panel.hidden = false;
  const content: (Node | string)[] = [];
  if (!s) {
    content.push(element("h2", "", "Private preview"), element("p", "", problem));
    panel.replaceChildren(...content);
    return;
  }
  const title = element("h2", "", `Private preview: ${s.citation}`);
  const reduce = reduceButton(panel, title);
  content.push(title);
  if (s.status === "received" || s.status === "processing") {
    const steps = s.steps.length ? s.steps : Object.keys(STEP_TEXT);
    const current = s.status === "received" ? -1 : steps.indexOf(s.step ?? "");
    const bar = element("progress");
    bar.max = steps.length;
    bar.value = Math.max(0, current + 0.5);
    const list = element("ol", "steps");
    steps.forEach((step, i) => {
      list.append(element("li", i < current ? "done" : i === current ? "current" : "", STEP_TEXT[step] ?? step));
    });
    content.push(
      element(
        "p",
        "",
        s.status === "received"
          ? s.waiting
            ? `Waiting: ${s.waiting} submission${s.waiting > 1 ? "s" : ""} to compile before yours.`
            : "Starting…"
          : "Your dataset is being prepared, usually in a few minutes. This page shows it once ready.",
      ),
      bar,
      list,
    );
  } else if (s.status === "ready") {
    content.push(
      element(
        "p",
        "",
        `${s.dataset}, compiled from your submission. Only people with this link can see it; it is removed once the review is done.`,
      ),
    );
    if (s.warnings?.length)
      content.push(issues(`The automatic checks found ${s.warnings.length} thing${s.warnings.length > 1 ? "s" : ""} to look at`, s.warnings));
    else content.push(element("p", "muted", "The automatic checks found no problem."));
  } else if (s.status === "failed") {
    content.push(
      element("p", "", "Your dataset could not be compiled automatically. We were told, and will come back to you."),
    );
    if (s.errors?.length) content.push(issues("Problems found in the files", s.errors, true));
  } else {
    content.push(element("p", "", "This preview has ended: the review of the submission is done."));
  }
  panel.replaceChildren(reduce, ...content);
}

/** The preview is closed (reviewed, failed, or a wrong link): a page that says so instead. */
function ended(status: string): void {
  location.replace(`/preview-ended/?status=${encodeURIComponent(status)}`);
}

/**
 * Show the progress in the panel until the preview is ready; its status then. When the preview
 * cannot be shown (failed, closed, wrong link), goes to /preview-ended/ instead.
 */
export function startPreview(panel: HTMLElement): Promise<PreviewStatus> {
  document.title = "Private preview · AntADatabase 3D viewer";
  const robots = document.createElement("meta");
  robots.name = "robots";
  robots.content = "noindex";
  document.head.append(robots);
  return new Promise((resolve) => {
    const poll = async () => {
      let status: PreviewStatus;
      try {
        const response = await fetch(`${SUBMISSION!.url}/status`, { cache: "no-store" });
        if (response.status === 404) {
          ended("missing");
          return;
        }
        if (!response.ok) throw new Error(`${response.status}`);
        status = await response.json();
      } catch {
        show(panel, null, "The portal cannot be reached at the moment: trying again…");
        setTimeout(poll, 4 * POLL_MS);
        return;
      }
      if (!["received", "processing", "ready"].includes(status.status)) {
        ended(status.status);
        return;
      }
      show(panel, status);
      if (status.status === "ready") resolve(status);
      else setTimeout(poll, POLL_MS);
    };
    poll();
  });
}
