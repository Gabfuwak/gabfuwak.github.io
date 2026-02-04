struct Uniforms {
  mvp : mat4x4<f32>,
  nbLights: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  lights: array<f32, 12>,
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
  var output_color = vec3f(0.0);

  if (u32(uniforms.nbLights) == 0){ // if there are no lights, don't render
    return vec4f(output_color, 1.0);
  }

  for(var i = 0u; i < u32(uniforms.nbLights); i++){
    let lightPos = vec3f(
      uniforms.lights[i * 3 + 0],
      uniforms.lights[i * 3 + 1],
      uniforms.lights[i * 3 + 2]
    );
    let lightDir = normalize(lightPos - input.worldPos.xyz);
    let lambertFactor = max(0.0, dot(lightDir, normalize(input.normal.xyz)));
    output_color += input.color.xyz * lambertFactor;
  }

  output_color /= f32(uniforms.nbLights);
  return vec4f(output_color, 1.0);
}
