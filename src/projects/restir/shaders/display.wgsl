// === display.wgsl ===
// Reads from the accumulation buffer and displays the averaged result.

struct PointLight {
  position:  vec3f, intensity: f32,
  color:     vec3f, _pad:      f32,
  direction: vec3f, angle_rad: f32,
};

struct Material {
  baseColor: vec3f, roughness: f32,
  fresnel:   vec3f, metalness: f32,
  emission:  f32,   _pad1: f32, _pad2: f32, _pad3: f32,
};

struct Uniforms {
  camera_pos:      vec3f, fov_factor:    f32,
  camera_forward:  vec3f, aspect_ratio:  f32,
  camera_right:    vec3f, nb_lights:     f32,
  camera_up:       vec3f, frame_count:   f32,
  lights:          array<PointLight, 4>,
  nb_materials:    f32,
  bvh_vis_depth:   f32,
  bvh_heat_max:    f32,
  bvh_early_stop:  f32,
  screen_width:    f32,
  screen_height:   f32,
  render_mode:     f32,
  _pad2:           f32,
  materials:       array<Material, 16>,
};

@group(0) @binding(0) var<storage, read> accum: array<vec4f>;
@group(0) @binding(1) var<uniform> u: Uniforms;

struct VertexOutput {
  @builtin(position) clip_pos:   vec4f,
  @location(0)       screen_pos: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let x = f32(i32(vi) / 2) * 4.0 - 1.0;
  let y = f32(i32(vi) % 2) * 4.0 - 1.0;
  var out: VertexOutput;
  out.clip_pos   = vec4f(x, y, 0.0, 1.0);
  out.screen_pos = vec2f(x, y);
  return out;
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f {
  let px = vec2u(input.clip_pos.xy);
  let pixel_value = accum[px.y * u32(u.screen_width) + px.x];
  return vec4f(pixel_value.rgb / pixel_value.w, 1.0);
}
