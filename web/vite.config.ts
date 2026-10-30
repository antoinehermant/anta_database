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

export default defineConfig({
  plugins: [goatcounter],
  // The submission portal (portal/app.py) in development
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
  build: {
    rollupOptions: {
      input: {
        home: page("index.html"),
        viewer: page("viewer/index.html"),
        download: page("download/index.html"),
        submit: page("submit/index.html"),
        contact: page("contact/index.html"),
        "preview-ended": page("preview-ended/index.html"),
        privacy: page("privacy/index.html"),
      },
    },
  },
});
