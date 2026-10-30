# AntADatabase 3D viewer

Static web app (Vite + TypeScript + three.js) showing Antarctica in 3D.

## Development

1. Export the Bedmap3 terrain once (about 30 s, writes `web/public/data/terrain/`, ~25 MB):
   ```bash
   python -m anta_database.web.terrain /path/to/bedmap3.nc -o web/public/data/terrain
   ```
2. Export the isochrones and their surface tracks (writes `web/public/data/irh/`, ~13 MB):
   ```bash
   python -m anta_database.web.irh /path/to/AntADatabase --bedmap /path/to/bedmap3.nc -o web/public/data/irh
   ```
3. Export the MEaSUREs ice velocities (NSIDC-0484 v2, 450 m; download needs a NASA Earthdata
   login; writes `web/public/data/velocity/`, ~4 MB):
   ```bash
   python -m anta_database.web.velocity antarctica_ice_velocity_450m_v2.nc -o web/public/data/velocity
   ```
4. Export the catalogue of the download page (descriptions and coverage maps of the datasets,
   a few seconds, writes `web/public/data/catalog/`):
   ```bash
   python -m anta_database.web.catalog /path/to/AntADatabase -o web/public/data/catalog
   ```
5. Run the dev server and open http://localhost:5173 (front page; viewer: http://localhost:5173/viewer/,
   download page: http://localhost:5173/download/, submit page: http://localhost:5173/submit/):
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
- Colours follow the light or dark mode of the browser.

## Layers loaded on demand

To keep the first load light, some data is only fetched when first needed:
- Ice velocity: a 2 km speed grid colours the ice surface (log scale), and 4 km velocity
  components move the flow lines (CPU advection, at most 30 updates per second while animated).
- Surface tracks: radar lines at the ice surface where IRHs are traced, with IRH_NUM and the
  oldest IRH age in the tooltip.
- Isochrone depths and fractional depths (IRH_DEPTH / ICE_THK, with the Bedmap3 thickness where
  ICE_THK is missing), for colouring by them. Colouring by elevation uses the line vertices, and
  no other geometry is loaded: only the colours of the existing lines are rewritten.

Filters (datasets, ages) hide lines with a per-segment flag read in the vertex shader, so the
geometry is never rebuilt.

## Download page

`download/index.html` (`src/download.ts`) lists the datasets and downloads the chosen ones in one
zip file. With all datasets, the zip holds the whole database in an `AntADatabase/` directory,
with the SQLite index and the manifest; with a selection, only the dataset directories, to put in
an existing database directory. The files come from the manifest of the
published database (`VITE_DB_URL`, the bucket by default, also in development), so the download
always matches what `anta_database.sync()` gets; only the descriptions come from `catalog.json`.
The zip is built in the browser while the files arrive, without compression (the HDF5 files gain
little). In Chromium browsers it is written straight to disk; elsewhere it is built in memory, then
saved. A selection is then indexed with `Database("path/to/AntADatabase", index_database=True)`, as
its README.txt explains.

## Deployment

Everything is on GitHub Pages, published by `./publish_site.sh` at the root of the repo
(`--no-push` to only build):

| Address | Content | Published to |
|---|---|---|
| https://antadatabase.com | `web/` (front page, `/viewer/`, `/download/`, `/submit/`) | `gh-pages` of the remote `site` |
| https://docs.antadatabase.com | the Jupyter Book (`book/`) | `gh-pages` of this repository |
| https://3d-antadatabase.com | redirect to the new site (`redirect.html`) | `gh-pages` of the remote `viewer-site` |

The old addresses keep working: 3d-antadatabase.com redirects every path to `/viewer/` (or
`/download/`), and the old copy of the viewer in the docs (`/viewer/`) too, keeping `#view=...`
so shared views still open. GitHub itself redirects antoinehermant.github.io/anta_database/ to
docs.antadatabase.com once the docs have their custom domain. Online, the viewer reads its data
from the public bucket (`VIEWER_DATA_URL` in the script), not from GitHub.

Setting up the domains, once:
1. Register `antadatabase.com` (and `antadatabase.org`, as a web redirect to the .com).
2. DNS of `antadatabase.com`: `A` records of the apex to GitHub Pages (185.199.108.153,
   185.199.109.153, 185.199.110.153, 185.199.111.153), `CNAME www` to `antoinehermant.github.io`,
   and `CNAME docs` to `antoinehermant.github.io`. Keep the DNS of `3d-antadatabase.com` as it is.
3. Create the GitHub repository of the site, `antoinehermant/antadatabase.com` (empty, public), and
   add it: `git remote add site git@github.com:antoinehermant/antadatabase.com.git`. The old
   repository `antoinehermant/3d-antadatabase` (remote `viewer-site`) now only holds the redirect
   of 3d-antadatabase.com; it can be deleted when that domain is given up.
4. Run `./publish_site.sh`, then in the Pages settings of each repository check the custom domain
   and tick "Enforce HTTPS" once the certificate is ready (a few minutes to an hour).
5. Optionally verify the domain in the GitHub account settings (Pages), and add the site to
   Google Search Console with `https://antadatabase.com/sitemap.xml`.

After re-exporting the data, upload it to the bucket (with the `infomaniak` AWS profile); the
site does not need to be republished:
```bash
aws --endpoint-url=https://s3.pub1.infomaniak.cloud s3 sync web/public/data s3://antadatabase/web/
```

The bucket must allow other sites to read it (CORS), once:
```bash
swift post antadatabase -H "X-Container-Meta-Access-Control-Allow-Origin: *"
```
