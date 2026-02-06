//   struct Uniforms { time: f32, resolution: vec2f }
//   @group(0) @binding(0) var<uniform> u: Uniforms;
//   struct FragIn { @builtin(position) position: vec4f, @location(0) uv: vec2f }
//
// Write your fragment shader below.
// Available: in.uv (0->1), u.time (seconds), u.resolution (pixels)

fn rotateY(angle: f32) -> mat3x3f {
  let c = cos(angle); let s = sin(angle);
  return mat3x3f(c, 0, s,
                 0, 1, 0,
                -s, 0, c);
}

fn rotateX(angle: f32) -> mat3x3f {
  let c = cos(angle); let s = sin(angle);
  return mat3x3f(1, 0, 0,
                 0, c, -s,
                 0, s, c);
}

fn sdfBox(point: vec3f, halfSize: vec3f) -> f32 {
  let offset = abs(point) - halfSize;
  return length(max(offset, vec3f(0.0))) + min(max(offset.x, max(offset.y, offset.z)), 0.0);
}

fn scene(point: vec3f) -> f32 {
  let rotated = rotateX(u.time * 0.7) * rotateY(u.time * 0.5) * point;
  return sdfBox(rotated, vec3f(0.5));
}

fn estimateNormal(point: vec3f) -> vec3f {
  let eps = vec2f(0.001, 0.0);
  return normalize(vec3f(
    scene(point + eps.xyy) - scene(point - eps.xyy),
    scene(point + eps.yxy) - scene(point - eps.yxy),
    scene(point + eps.yyx) - scene(point - eps.yyx),
  ));
}

@fragment
fn fragmentMain(in: FragIn) -> @location(0) vec4f {
  var uv = (in.uv - 0.5) * 2.0;
  uv.x *= u.resolution.x / u.resolution.y;

  let rayOrigin = vec3f(0.0, 0.0, -4.0);
  let rayDir = normalize(vec3f(uv, 3.5));

  var dist = 0.0;
  var hit = false;
  for (var step = 0; step < 64; step++) {
    let pos = rayOrigin + rayDir * dist;
    let closest = scene(pos);
    if (closest < 0.001) { hit = true; break; }
    if (dist > 10.0) { break; }
    dist += closest;
  }

  let background = vec3f(1.0, 0.98, 0.96);
  if (!hit) { return vec4f(background, 1.0); }

  let hitPoint = rayOrigin + rayDir * dist;
  let normal = estimateNormal(hitPoint);
  let lightDir = normalize(vec3f(1.0, 1.0, -1.0));
  let diffuse = max(dot(normal, lightDir), 0.0);
  let ambient = 0.15;

  let lavender = vec3f(0.45, 0.53, 0.99);
  let color = lavender * (diffuse + ambient);
  return vec4f(color, 1.0);
}

