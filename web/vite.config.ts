import { resolve } from "node:path";
import { defineConfig } from "vite";

// The pages of antadatabase.com: front page, 3D viewer (/viewer/), download (/download/),
// submission (/submit/), contact (/contact/), and where closed
// previews of submissions go (/preview-ended/)
const page = (path: string) => resolve(import.meta.dirname, path);

export default defineConfig({
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
      },
    },
  },
});
