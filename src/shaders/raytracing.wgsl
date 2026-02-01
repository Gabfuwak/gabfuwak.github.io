struct Uniforms {
  mvp : mat4x4<f32>,
  lightPos: vec3f,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;


struct VertexInput {
  @location(0) pos: vec3f,
  @location(1) color: vec3f,
  @location(2) normal: vec3f,
  @builtin(instance_index) instance: u32,
};


struct VertexOutput {
  @builtin(position) pos : vec4f,
  @location(0) color: vec4f,
  @location(1) normal: vec4f,
  @location(2) worldPos: vec4f,
};

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.pos = uniforms.mvp * vec4f(input.pos, 1);
  output.worldPos = vec4f(input.pos, 1);
  output.color = vec4f(input.color, 1);
  output.normal = vec4f(input.normal, 0);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let lightDir: vec3f =  normalize(uniforms.lightPos - input.worldPos.xyz);
  let lambertFactor: f32 = max(0.0, dot(lightDir, normalize(input.normal.xyz)));
  return input.color * lambertFactor;
}
