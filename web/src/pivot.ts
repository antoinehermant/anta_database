/**
 * Rotation centre of the camera: the middle of the ice column (or the bed
 * where there is no ice, but not below sea level) under a point of the screen.
 */
import * as THREE from "three";
import type { Grid } from "./terrain";

export class PivotField {
  private constructor(
    private grid: Grid,
    private heights: Float32Array, // m
    private bed: Float32Array, // m, NaN without data
    private surface: Float32Array, // m, NaN without ice
  ) {}

  /** From the bed and surface heights and validities of the same grid. */
  static fromLayers(
    grid: Grid,
    bed: { heights: Float32Array; valid: Uint8Array },
    surface: { heights: Float32Array; valid: Uint8Array },
  ): PivotField {
    const heights = new Float32Array(bed.heights.length);
    const bedHeights = new Float32Array(bed.heights.length);
    const surfaceHeights = new Float32Array(bed.heights.length);
    for (let i = 0; i < heights.length; i++) {
      bedHeights[i] = bed.valid[i] ? bed.heights[i] : NaN;
      surfaceHeights[i] = surface.valid[i] ? surface.heights[i] : NaN;
      if (surface.valid[i]) heights[i] = (surface.heights[i] + bed.heights[i]) / 2;
      else if (bed.valid[i]) heights[i] = Math.max(bed.heights[i], 0);
    }
    return new PivotField(grid, heights, bedHeights, surfaceHeights);
  }

  /** Bed elevation (m) at a point (km), null outside the data. */
  bedAt(x: number, y: number): number | null {
    return this.valueAt(this.bed, x, y);
  }

  /** Ice surface elevation (m) at a point (km), null where there is no ice. */
  surfaceAt(x: number, y: number): number | null {
    return this.valueAt(this.surface, x, y);
  }

  private valueAt(values: Float32Array, x: number, y: number): number | null {
    const { nx, ny, dx, x0, y0 } = this.grid;
    const i = Math.round((x - x0) / dx);
    const j = Math.round((y - y0) / dx);
    if (i < 0 || i >= nx || j < 0 || j >= ny) return null;
    const h = values[j * nx + i];
    return Number.isNaN(h) ? null : h;
  }

  /** Height (m) at a point (km), 0 outside the grid. */
  heightAt(x: number, y: number): number {
    const { nx, ny, dx, x0, y0 } = this.grid;
    const i = Math.round((x - x0) / dx);
    const j = Math.round((y - y0) / dx);
    if (i < 0 || i >= nx || j < 0 || j >= ny) return 0;
    return this.heights[j * nx + i];
  }

  /**
   * First point where a ray (scene units: km, z exaggerated) goes below the
   * field, or null if it doesn't within maxDistance.
   */
  intersect(origin: THREE.Vector3, direction: THREE.Vector3, exaggeration: number, maxDistance: number) {
    const scale = exaggeration / 1000;
    const below = (t: number) => {
      const x = origin.x + t * direction.x;
      const y = origin.y + t * direction.y;
      return origin.z + t * direction.z <= this.heightAt(x, y) * scale;
    };
    if (below(0)) return null; // the camera itself is below: keep the current centre
    let previous = 0;
    for (let t = 1; t <= maxDistance; t += Math.max(1, t * 0.002)) {
      if (below(t)) {
        // Refine between the last point above and the first point below
        let lo = previous;
        let hi = t;
        for (let k = 0; k < 20; k++) {
          const mid = (lo + hi) / 2;
          if (below(mid)) hi = mid;
          else lo = mid;
        }
        return origin.clone().addScaledVector(direction, hi);
      }
      previous = t;
    }
    return null;
  }
}
