/**
 * What is in the way of the open profile: its curtain (from the bed to the ice surface along
 * its line) is drawn into a depth texture from the camera, before each frame. The lines then
 * fade where they are drawn over the curtain, and closer to the camera than it (LineLayer).
 */
import * as THREE from "three";

export class InTheWay {
  // Shared with the materials of the lines
  readonly uniforms = {
    inWayDepth: { value: null as THREE.Texture | null },
    inWayViewport: { value: new THREE.Vector2(1, 1) },
    inWayActive: { value: 0 },
  };
  private scene = new THREE.Scene();
  private group = new THREE.Group();
  private target = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: true });
  private material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, colorWrite: false });
  private mesh: THREE.Mesh | null = null;
  private size = new THREE.Vector2();

  constructor() {
    this.target.depthTexture = new THREE.DepthTexture(1, 1);
    this.uniforms.inWayDepth.value = this.target.depthTexture;
    this.scene.add(this.group);
  }

  setExaggeration(exaggeration: number): void {
    this.group.scale.z = exaggeration / 1000;
  }

  /** The curtain of a line: x, y (km), its top and bottom (m) at each point; null for none. */
  setCurtain(curtain: { x: Float32Array; y: Float32Array; top: Float32Array; bottom: Float32Array } | null): void {
    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.group.remove(this.mesh);
      this.mesh = null;
    }
    this.uniforms.inWayActive.value = 0;
    if (!curtain) return;
    const { x, y, top, bottom } = curtain;
    const ok = (i: number) => Number.isFinite(x[i]) && Number.isFinite(y[i]) && Number.isFinite(top[i]) && Number.isFinite(bottom[i]);
    const positions = new Float32Array(x.length * 6);
    const index: number[] = [];
    for (let i = 0; i < x.length; i++) {
      positions.set([x[i], y[i], ok(i) ? bottom[i] : 0, x[i], y[i], ok(i) ? top[i] : 0], i * 6);
      // A quad to the next point, unless either has no data (between the parts of the line)
      if (i + 1 < x.length && ok(i) && ok(i + 1)) index.push(2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 1, 2 * i + 3, 2 * i + 2);
    }
    if (!index.length) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(index);
    this.mesh = new THREE.Mesh(geometry, this.material);
    this.mesh.frustumCulled = false;
    this.group.add(this.mesh);
    this.uniforms.inWayActive.value = 1;
  }

  /** Draw the depth of the curtain from the camera, at the size of the frame. Call before each frame. */
  render(renderer: THREE.WebGLRenderer, camera: THREE.Camera): void {
    if (!this.mesh) return;
    renderer.getDrawingBufferSize(this.size);
    if (this.target.width !== this.size.x || this.target.height !== this.size.y) this.target.setSize(this.size.x, this.size.y);
    this.uniforms.inWayViewport.value.copy(this.size);
    const previous = renderer.getRenderTarget();
    renderer.setRenderTarget(this.target);
    renderer.clear(false, true, false);
    renderer.render(this.scene, camera);
    renderer.setRenderTarget(previous);
  }
}
