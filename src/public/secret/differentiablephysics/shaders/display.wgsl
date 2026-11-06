// === display.wgsl ===

struct PointLight {
  position:  vec3f, intensity: f32,
  color:     vec3f, _pad:      f32,
};

struct Material {
  baseColor: vec3f, roughness: f32,
  fresnel:   vec3f, metalness: f32,
};

struct Uniforms {
  camera_pos:      vec3f, fov_factor:    f32,
  camera_forward:  vec3f, aspect_ratio:  f32,
  camera_right:    vec3f, nb_lights:     f32,
  camera_up:       vec3f, _pad_camera:   f32,
  lights:          array<PointLight, 4>,
  screen_width:    f32,
  screen_height:   f32,
  _pad_screen:     vec2f,
  materials:       array<Material, 16>,
  floor_height:    f32,
  floor_material:  u32,
  visible:         u32,
  _pad_display:    u32,
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
  return vec4f(pixel_value.rgb, 1.0);
}
