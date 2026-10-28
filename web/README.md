# AntADatabase 3D viewer

Static web app (Vite + TypeScript + three.js) showing Antarctica in 3D.

## Development

1. Export the Bedmap3 terrain once (about 30 s, writes `web/public/data/terrain/`, ~25 MB):
   ```bash
   python -m anta_database.web.terrain /path/to/bedmap3.nc -o web/public/data/terrain
   ```
2. Run the dev server and open http://localhost:5173:
   ```bash
   cd web
   npm install
   npm run dev
   ```

`npm run build` writes the static site to `web/dist/`.

## How the terrain is drawn

- The 500 m Bedmap3 grids are block-averaged to 2, 4 and 8 km int16 heightmaps.
- The viewer first shows the 8 km preview, then a 4 km mesh shaded per pixel from the 2 km heightmap.
- Heights are applied in the vertex shader, so the vertical exaggeration is a single uniform.
- Lighting is computed in the fragment shader from the finer texture, so the relief looks sharper than the mesh.
- The camera view is kept in the URL (`#view=...`), so views can be shared.
