// Per-pixel hillshading from the height texture, so the relief is sharper
// than the mesh, and follows the vertical exaggeration.
uniform sampler2D heightTex;
uniform sampler2D maskTex;
uniform sampler2D colormapTex;
uniform vec2 colormapRange; // m
uniform vec2 texSize;
uniform float texDx; // km
uniform float exaggeration;
uniform vec3 lightDir;
uniform float ambient;
uniform vec3 baseColor;
uniform vec3 shelfColor;
uniform float opacity;

varying vec2 vUv;

float heightAt(vec2 uv) {
  return texture(heightTex, uv).r;
}

void main() {
  // Validity per pixel from the texture, smoother than per mesh vertex
  if (texture(heightTex, vUv).g < 0.5) discard;

  float h = heightAt(vUv);
  vec2 t = 1.0 / texSize;
  float dzdx = heightAt(vUv + vec2(t.x, 0.0)) - heightAt(vUv - vec2(t.x, 0.0));
  float dzdy = heightAt(vUv + vec2(0.0, t.y)) - heightAt(vUv - vec2(0.0, t.y));
  float scale = exaggeration / (2.0 * texDx * 1000.0);
  vec3 normal = normalize(vec3(-dzdx * scale, -dzdy * scale, 1.0));
  float diffuse = max(dot(normal, normalize(lightDir)), 0.0);

#if USE_COLORMAP
  float c = (h - colormapRange.x) / (colormapRange.y - colormapRange.x);
  vec3 color = texture(colormapTex, vec2(clamp(c, 0.0, 1.0), 0.5)).rgb;
#else
  float mask = texture(maskTex, vUv).r * 255.0;
  vec3 color = (mask > 1.5 && mask < 3.5) ? shelfColor : baseColor;
#endif

  gl_FragColor = vec4(color * (ambient + (1.0 - ambient) * diffuse), opacity);
  #include <colorspace_fragment>
}
