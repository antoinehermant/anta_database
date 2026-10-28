// Displaces a flat grid (km, EPSG:3031) by the height texture (m).
uniform sampler2D heightTex;
uniform vec2 texOrigin; // centre of the first texel, km
uniform float texDx; // texel size, km
uniform vec2 texSize; // texels
uniform float exaggeration;

varying vec2 vUv;

void main() {
  vUv = (position.xy - texOrigin) / (texDx * texSize) + 0.5 / texSize;
  float h = texture(heightTex, vUv).r;
  vec3 p = vec3(position.xy, h * exaggeration / 1000.0);
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
}
