struct VertexOutput {
  @builtin(position) clip_pos: vec4f,
  @location(0) screen_pos: vec2f,
};

struct RayUniforms {
  camera_pos: vec3f,
  fov_factor: f32,          // uses camera_pos padding slot
  camera_forward: vec3f,
  aspect_ratio: f32,        // uses camera_forward padding slot
  camera_right: vec3f,
  _pad0: f32,               // unused padding
  camera_up: vec3f,
  _pad1: f32,               // unused padding
  lightPos: vec3f,
  _pad2: f32,               // unused padding
};

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>
};


@group(0) @binding(0)
var<uniform> uniforms : RayUniforms;

fn ray_at(screenCoord: vec2<f32>) -> Ray {
  var output: Ray;

  let horizontal = screenCoord.x * uniforms.fov_factor * uniforms.aspect_ratio;
  let vertical = screenCoord.y * uniforms.fov_factor;

  output.origin = uniforms.camera_pos;
  output.direction = normalize(
    uniforms.camera_forward +
    horizontal * uniforms.camera_right +
    vertical * uniforms.camera_up
  );

  return output;
}

@vertex
fn vertexMain(@location(0) pos: vec2f) -> VertexOutput {
  var output: VertexOutput;
  output.clip_pos = vec4f(pos, 0.0, 1.0);
  output.screen_pos = pos;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  // debug viz for the ray_at function
  var output: Ray = ray_at(input.screen_pos);
  var c: f32 = dot(output.direction, uniforms.camera_forward);
  let difference = (1.0 - c) * 10.0;
  return vec4f(difference, difference, difference, 1.0);
}
