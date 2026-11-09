import { resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

// The pages of antadatabase.com: front page, 3D viewer (/viewer/), download (/download/),
// submission (/submit/), contact (/contact/), privacy (/privacy/), and where closed
// previews of submissions go (/preview-ended/)
const page = (path: string) => resolve(import.meta.dirname, path);

// Visitor counts (GoatCounter: no cookies, no personal data kept), on the published site only.
// Not on the private previews of submissions: their key is in the address, which the counter
// sends with each visit
const GOATCOUNTER = "https://antadatabase.goatcounter.com/count";
const goatcounter: Plugin = {
  name: "goatcounter",
  apply: "build",
  transformIndexHtml: () => [
    {
      tag: "script",
      injectTo: "head",
      children: `if (!new URLSearchParams(location.search).has("key")) {
        const s = document.createElement("script");
        s.async = true;
        s.src = "https://gc.zgo.at/count.js";
        s.dataset.goatcounter = ${JSON.stringify(GOATCOUNTER)};
        document.head.append(s);
      }`,
    },
  ],
};

// The development viewer (VITE_DEV_VIEWER, built by publish_site.sh): the next version of the
// data, at an address not linked, not indexed, and saying what it is
const devViewer: Plugin = {
  name: "dev-viewer",
  apply: "build",
  transformIndexHtml: () =>
    process.env.VITE_DEV_VIEWER
      ? [
          { tag: "meta", attrs: { name: "robots", content: "noindex, nofollow" }, injectTo: "head" },
          {
            tag: "div",
            attrs: {
              id: "dev-banner",
              style:
                "position:fixed;top:0;left:50%;transform:translateX(-50%);z-index:50;padding:3px 12px;" +
                "border-radius:0 0 8px 8px;background:#b5470b;color:#fff;font:600 0.8rem system-ui,sans-serif",
            },
            children: 'Development version: data v2, not published. <a href="/viewer/" style="color:#fff">Official viewer</a>',
            injectTo: "body-prepend",
          },
        ]
      : [],
};

export default defineConfig({
  plugins: [goatcounter, devViewer],
  // The submission portal (portal/app.py) in development
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
  build: {
    rollupOptions: {
      input: {
        home: page("index.html"),
        viewer: page("viewer/index.html"),
        lesson: page("lesson/index.html"),
        download: page("download/index.html"),
        submit: page("submit/index.html"),
        contact: page("contact/index.html"),
        "preview-ended": page("preview-ended/index.html"),
        privacy: page("privacy/index.html"),
      },
    },
  },
});
