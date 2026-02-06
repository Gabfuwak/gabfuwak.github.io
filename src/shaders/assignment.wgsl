struct PointLight{
  position: vec3<f32>,
  _pad0: f32, // will be intensity later
  color: vec3<f32>,
  _pad1: f32,
  // will add color later
  // will be an other padding later
}

struct Material{
  baseColor: vec3<f32>,
  _pad0: f32
}

struct Uniforms {
  mvp : mat4x4<f32>,
  nbLights: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
  lights: array<PointLight, 4>,
  nbMaterials: f32,
  _pad4: f32,
  _pad5: f32,
  _pad6: f32,
  materials: array<Material, 16>,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;


struct VertexInput {
  @location(0) pos: vec3f,
  @location(1) objectId: u32,
  @location(2) normal: vec3f,
  @builtin(instance_index) instance: u32,
};


struct VertexOutput {
  @builtin(position) pos : vec4f,
  @location(0) @interpolate(flat) objectId: u32,
  @location(1) normal: vec4f,
  @location(2) worldPos: vec4f,
};

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.pos = uniforms.mvp * vec4f(input.pos, 1);
  output.worldPos = vec4f(input.pos, 1);
  output.objectId = input.objectId;
  output.normal = vec4f(input.normal, 0);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var output_color = vec3f(0.0);

  if (u32(uniforms.nbLights) == 0){ // if there are no lights, don't render
    return vec4f(output_color, 1.0);
  }

  // Look up material by objectId
  let material = uniforms.materials[input.objectId];
  let baseColor = material.baseColor;

  for(var i = 0u; i < u32(uniforms.nbLights); i++){
    let lightDir = normalize(uniforms.lights[i].position - input.worldPos.xyz);
    let lambertFactor = max(0.0, dot(lightDir, normalize(input.normal.xyz)));
    output_color += baseColor * lambertFactor * uniforms.lights[i].color;
  }

  output_color /= f32(uniforms.nbLights);
  return vec4f(output_color, 1.0);
}
