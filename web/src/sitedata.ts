/**
 * Exported viewer data shared by the pages around the viewer (download, submit): the
 * catalogue of the datasets and the Bedmap3 mask as a map background.
 */

// Exported viewer data (catalogue, Bedmap3 mask), as in the viewer
export const DATA_ROOT = (import.meta.env.VITE_DATA_URL || `${import.meta.env.BASE_URL}data`).replace(/\/$/, "");

// Exported files may change: always revalidate cached copies (as in the viewer)
export const fetchData = (url: string) => fetch(url, { cache: "no-cache" });

export async function fetchGzip(url: string): Promise<ArrayBuffer> {
  const response = await fetchData(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  const buffer = await response.arrayBuffer();
  const bytes = new Uint8Array(buffer, 0, 2);
  // Servers may already have decompressed it (Content-Encoding: gzip)
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return buffer;
  return new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}

export interface Irh {
  irh: number;
  age: number | null;
  age_unc: number | null;
}

export interface CatalogDataset {
  name: string;
  citation: string;
  doi_dataset: string[];
  doi_publication: string[];
  institutes: string[];
  projects: string[];
  radar_instruments: string[];
  acquisition_years: string;
  regions: string[];
  imbie_basins: string[];
  n_lines: number;
  n_points: number;
  irhs: Irh[];
  variables: string[];
  coverage: number[];
}

export interface Catalog {
  coverage_grid: { cell_km: number; x0_km: number; y0_km: number; n: number };
  datasets: CatalogDataset[];
}

export async function loadCatalog(): Promise<Catalog> {
  const response = await fetchData(`${DATA_ROOT}/catalog/catalog.json`);
  if (!response.ok) throw new Error(`Catalogue: ${response.status} ${response.statusText}`);
  return response.json();
}

/** Bedmap3 mask in the colours of the theme, ±3333 km in EPSG:3031 with north up, or null. */
export async function loadMask(): Promise<ImageBitmap | null> {
  try {
    const meta = await (await fetchData(`${DATA_ROOT}/terrain/terrain.json`)).json();
    const level = meta.resolutions["8km"];
    const mask = new Uint8Array(await fetchGzip(`${DATA_ROOT}/terrain/${level.layers.mask.file}`));
    const dark = document.documentElement.dataset.theme === "dark";
    const colors: [number, number, number, number][] = [
      [0, 0, 0, 0], // ocean
      dark ? [92, 108, 128, 255] : [218, 228, 238, 255], // grounded ice
      dark ? [70, 86, 106, 255] : [190, 205, 221, 255], // ice shelves
      dark ? [70, 86, 106, 255] : [190, 205, 221, 255],
      dark ? [120, 110, 100, 255] : [150, 140, 130, 255], // rock
    ];
    const { nx, ny } = level;
    const image = new ImageData(nx, ny);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        // Rows are y ascending: flip so north (y up) is at the top
        const c = colors[mask[j * nx + i]] ?? colors[0];
        image.data.set(c, ((ny - 1 - j) * nx + i) * 4);
      }
    }
    return await createImageBitmap(image);
  } catch (e) {
    console.warn("No map background:", e);
    return null;
  }
}

/** Half width of the map background, in km. */
export const MAP_EXTENT_KM = 3333;
