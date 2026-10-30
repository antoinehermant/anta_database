/**
 * Moving the view without dragging: W A S D or the arrow keys move over the ground (forward,
 * left, back, right, as seen on screen), Q and E turn around the centre of the view. The same
 * moves are buttons of a pad in the bottom right corner, held down like the keys.
 *
 * The camera and the rotation centre (the target of the OrbitControls) move together, at a
 * speed that follows the distance between them, so it feels the same at any zoom.
 */
import * as THREE from "three";
import type { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type Move = "forward" | "back" | "left" | "right" | "turnLeft" | "turnRight";

const KEYS: Record<string, Move> = {
  KeyW: "forward",
  ArrowUp: "forward",
  KeyS: "back",
  ArrowDown: "back",
  KeyA: "left",
  ArrowLeft: "left",
  KeyD: "right",
  ArrowRight: "right",
  KeyQ: "turnLeft",
  KeyE: "turnRight",
};

// Per second: a share of the distance between the camera and the centre, and an angle
const SPEED = 0.6;
const TURN_SPEED = Math.PI / 4;

// Buttons of the pad: move, symbol, title, grid area
const PAD: [Move, string, string, string][] = [
  ["turnLeft", "⟲", "Turn left (Q)", "tl"],
  ["forward", "▲", "Forward (W or ↑)", "up"],
  ["turnRight", "⟳", "Turn right (E)", "tr"],
  ["left", "◀", "Left (A or ←)", "left"],
  ["back", "▼", "Back (S or ↓)", "down"],
  ["right", "▶", "Right (D or →)", "right"],
];

/** Typing in a field of the menu (a number, a list) is not moving. */
function typing(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return true;
  return target instanceof HTMLInputElement && !["checkbox", "radio", "button", "range"].includes(target.type);
}

export class KeyNavigation {
  private held = new Set<Move>();
  private forward = new THREE.Vector3();
  private right = new THREE.Vector3();
  private offset = new THREE.Vector3();
  private readonly up = new THREE.Vector3(0, 0, 1);

  /**
   * camera, controls: the view to move. onEnd: called when a move ends (the view in the URL).
   */
  constructor(
    private camera: THREE.PerspectiveCamera,
    private controls: OrbitControls,
    private onEnd: () => void,
  ) {
    window.addEventListener("keydown", (event) => {
      const move = KEYS[event.code];
      if (!move || event.ctrlKey || event.metaKey || event.altKey || typing(event.target)) return;
      event.preventDefault(); // no scrolling with the arrows
      this.held.add(move);
    });
    window.addEventListener("keyup", (event) => {
      const move = KEYS[event.code];
      if (move) this.release(move);
    });
    // Keys released in another window would stay held
    window.addEventListener("blur", () => this.releaseAll());
  }

  /** The pad of buttons, in container. */
  addPad(container: HTMLElement): HTMLElement {
    const pad = document.createElement("div");
    pad.id = "move-pad";
    pad.setAttribute("aria-label", "Move the view");
    for (const [move, symbol, title, area] of PAD) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = symbol;
      button.title = title;
      button.setAttribute("aria-label", title);
      button.style.gridArea = area;
      if (move.startsWith("turn")) button.className = "turn";
      button.addEventListener("pointerdown", (event) => {
        button.setPointerCapture(event.pointerId);
        this.held.add(move);
      });
      for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
        button.addEventListener(type, () => this.release(move));
      }
      // Not a drag of the view, nor a text selection
      button.addEventListener("contextmenu", (event) => event.preventDefault());
      pad.append(button);
    }
    container.append(pad);
    return pad;
  }

  private release(move: Move): void {
    if (!this.held.delete(move)) return;
    if (!this.held.size) this.onEnd();
  }

  private releaseAll(): void {
    if (!this.held.size) return;
    this.held.clear();
    this.onEnd();
  }

  get active(): boolean {
    return this.held.size > 0;
  }

  /** Move the view for dt seconds of the held moves, before controls.update(). True if it moved. */
  update(dt: number): boolean {
    if (!this.held.size) return false;
    const { camera, controls, held } = this;
    // Forward: where the camera looks, on the ground; looking straight down, the top of the screen
    camera.getWorldDirection(this.forward).setZ(0);
    if (this.forward.lengthSq() < 1e-6) this.forward.set(0, 1, 0).applyQuaternion(camera.quaternion).setZ(0);
    this.forward.normalize();
    this.right.crossVectors(this.forward, this.up);

    const step = SPEED * camera.position.distanceTo(controls.target) * dt;
    const along = (held.has("forward") ? 1 : 0) - (held.has("back") ? 1 : 0);
    const across = (held.has("right") ? 1 : 0) - (held.has("left") ? 1 : 0);
    const turn = (held.has("turnLeft") ? 1 : 0) - (held.has("turnRight") ? 1 : 0);
    if (!along && !across && !turn) return false;

    const move = this.forward.multiplyScalar(along * step).addScaledVector(this.right, across * step);
    camera.position.add(move);
    controls.target.add(move);
    if (turn) {
      // Around the vertical through the centre of the view
      this.offset.subVectors(camera.position, controls.target).applyAxisAngle(this.up, turn * TURN_SPEED * dt);
      camera.position.copy(controls.target).add(this.offset);
    }
    return true;
  }
}
