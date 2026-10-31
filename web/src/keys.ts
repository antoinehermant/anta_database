/**
 * Moving the view without dragging: W A S D or the arrow keys move over the ground (forward,
 * left, back, right, as seen on screen), Q and E turn around the centre of the view, Z, I or +
 * zoom in, X, O or - zoom out, and 0 goes back to the default view. The same moves are buttons
 * of a pad in the bottom right corner, held down like the keys. Pointing at an edge of the view
 * also moves over the ground towards it (setEdge(), from the position of the mouse).
 *
 * The camera and the rotation centre (the target of the OrbitControls) move together, at a
 * speed that follows the distance between them, so it feels the same at any zoom.
 *
 * Letters are keys at a position (event.code: W A S D are Z Q S D on AZERTY), the + and - signs
 * are characters (event.key), wherever the layout puts them.
 */
import * as THREE from "three";
import type { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type Move = "forward" | "back" | "left" | "right" | "turnLeft" | "turnRight" | "zoomIn" | "zoomOut";

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
  KeyZ: "zoomIn",
  KeyI: "zoomIn",
  NumpadAdd: "zoomIn",
  KeyX: "zoomOut",
  KeyO: "zoomOut",
  NumpadSubtract: "zoomOut",
};

// Characters, on any key: "=" is + without shift on US and UK keyboards
const CHARACTERS: Record<string, Move> = { "+": "zoomIn", "=": "zoomIn", "-": "zoomOut", _: "zoomOut" };
const RESET_KEYS = ["Digit0", "Numpad0"];

// Per second: a share of the distance between the camera and the centre, and an angle
const SPEED = 0.6;
const TURN_SPEED = Math.PI / 4;
// Zooming divides the distance to the centre by e every 1 / ZOOM_SPEED seconds
const ZOOM_SPEED = 1.2;

// Buttons of the pad: move, symbol, title, grid area
const PAD: [Move, string, string, string][] = [
  ["turnLeft", "⟲", "Turn left (Q)", "tl"],
  ["forward", "▲", "Forward (W or ↑)", "up"],
  ["turnRight", "⟳", "Turn right (E)", "tr"],
  ["left", "◀", "Left (A or ←)", "left"],
  ["back", "▼", "Back (S or ↓)", "down"],
  ["right", "▶", "Right (D or →)", "right"],
  ["zoomIn", "+", "Zoom in (Z, I or +)", "in"],
  ["zoomOut", "−", "Zoom out (X, O or -)", "out"],
];

/** Typing in a field of the menu (a number, a list) is not moving. */
function typing(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return true;
  return target instanceof HTMLInputElement && !["checkbox", "radio", "button", "range"].includes(target.type);
}

export class KeyNavigation {
  private held = new Set<Move>();
  // Move of each key held down, released with the same key (the character can change with shift)
  private keys = new Map<string, Move>();
  private forward = new THREE.Vector3();
  private right = new THREE.Vector3();
  private offset = new THREE.Vector3();
  private readonly up = new THREE.Vector3(0, 0, 1);
  // Move towards an edge of the view: forward (top) and right, each from -1 to 1
  private edge = { along: 0, across: 0 };

  /**
   * camera, controls: the view to move. onEnd: called when a move ends (the view in the URL).
   * onReset: goes back to the default view.
   */
  constructor(
    private camera: THREE.PerspectiveCamera,
    private controls: OrbitControls,
    private onEnd: () => void,
    onReset: () => void,
  ) {
    window.addEventListener("keydown", (event) => {
      if (event.ctrlKey || event.metaKey || event.altKey || typing(event.target)) return;
      if (RESET_KEYS.includes(event.code)) {
        event.preventDefault();
        onReset();
        return;
      }
      const move = KEYS[event.code] ?? CHARACTERS[event.key];
      if (!move) return;
      event.preventDefault(); // no scrolling with the arrows
      this.keys.set(event.code, move);
      this.held.add(move);
    });
    window.addEventListener("keyup", (event) => {
      const move = this.keys.get(event.code);
      if (!move) return;
      this.keys.delete(event.code);
      // Another key may still hold the same move (Z and I)
      if (![...this.keys.values()].includes(move)) this.release(move);
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
      if (move.startsWith("zoom")) button.className = "zoom";
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
    this.keys.clear();
    this.setEdge(0, 0);
    if (!this.held.size) return;
    this.held.clear();
    this.onEnd();
  }

  get active(): boolean {
    return this.held.size > 0 || this.edgeActive;
  }

  private get edgeActive(): boolean {
    return this.edge.along !== 0 || this.edge.across !== 0;
  }

  /** Move towards an edge of the view, as fast as along and across (-1 to 1); 0, 0 stops. */
  setEdge(along: number, across: number): void {
    const was = this.edgeActive;
    this.edge = { along, across };
    if (was && !this.edgeActive && !this.held.size) this.onEnd();
  }

  /** Move the view for dt seconds of the held moves, before controls.update(). True if it moved. */
  update(dt: number): boolean {
    if (!this.held.size && !this.edgeActive) return false;
    const { camera, controls, held } = this;
    // Forward: where the camera looks, on the ground; looking straight down, the top of the screen
    camera.getWorldDirection(this.forward).setZ(0);
    if (this.forward.lengthSq() < 1e-6) this.forward.set(0, 1, 0).applyQuaternion(camera.quaternion).setZ(0);
    this.forward.normalize();
    this.right.crossVectors(this.forward, this.up);

    const step = SPEED * camera.position.distanceTo(controls.target) * dt;
    const along = THREE.MathUtils.clamp((held.has("forward") ? 1 : 0) - (held.has("back") ? 1 : 0) + this.edge.along, -1, 1);
    const across = THREE.MathUtils.clamp((held.has("right") ? 1 : 0) - (held.has("left") ? 1 : 0) + this.edge.across, -1, 1);
    const turn = (held.has("turnLeft") ? 1 : 0) - (held.has("turnRight") ? 1 : 0);
    const zoom = (held.has("zoomIn") ? 1 : 0) - (held.has("zoomOut") ? 1 : 0);
    if (!along && !across && !turn && !zoom) return false;

    const move = this.forward.multiplyScalar(along * step).addScaledVector(this.right, across * step);
    camera.position.add(move);
    controls.target.add(move);
    if (turn) {
      // Around the vertical through the centre of the view
      this.offset.subVectors(camera.position, controls.target).applyAxisAngle(this.up, turn * TURN_SPEED * dt);
      camera.position.copy(controls.target).add(this.offset);
    }
    if (zoom) {
      // Towards the centre of the view, within the distances allowed by the controls
      this.offset.subVectors(camera.position, controls.target);
      const distance = THREE.MathUtils.clamp(
        this.offset.length() * Math.exp(-zoom * ZOOM_SPEED * dt),
        controls.minDistance,
        controls.maxDistance,
      );
      camera.position.copy(controls.target).add(this.offset.setLength(distance));
    }
    return true;
  }
}
