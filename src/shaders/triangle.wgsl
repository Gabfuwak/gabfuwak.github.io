struct Uniforms {
  mvp : mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;


struct VertexInput {
  @location(0) pos: vec3f,
  @location(1) color: vec3f,
  @builtin(instance_index) instance: u32
};


struct VertexOutput {
  @builtin(position) pos : vec4f,
  @location(0) color: vec4f
};

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.pos = uniforms.mvp * vec4f(input.pos, 1);
  output.color = vec4f(input.color, 1);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
