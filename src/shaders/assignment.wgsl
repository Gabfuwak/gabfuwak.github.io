struct PointLight{
  position: vec3<f32>,
  intensity: f32,
  color: vec3<f32>,
  _pad1: f32,
  direction: vec3<f32>,
  angle: f32,
}

struct Material{
  baseColor: vec3<f32>,
  roughness: f32,
  fresnel: vec3<f32>,
  metalness: f32,
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


// ============================================================================
// Shared shading function - can be copy-pasted between shaders
// ============================================================================

fn evaluateRadiance(
  light: PointLight,
  worldPos: vec3<f32>,
  normal: vec3<f32>,
  material: Material
) -> vec3<f32> {
  let lightPosVector = light.position - worldPos;
  let lightDir = normalize(lightPosVector);

  // Check if point is within the light's cone
  let lightToPoint = -lightDir; // Direction from light to point
  let coneAngle = light.angle;
  let inCone = dot(normalize(light.direction), lightToPoint) >= cos(coneAngle / 2.0);

  if (!inCone) {
    return vec3<f32>(0.0); // Outside cone, no contribution
  }

  // Attenuation
  let lightDistance = length(lightPosVector) / 500.0;
  let constant = 1.0;
  let linear = 0.09;
  let quadratic = 0.032;
  let attenuationFactor = 1.0 / (constant + linear * lightDistance + quadratic * lightDistance * lightDistance);

  // Lambert shading
  let lambertFactor = max(0.0, dot(lightDir, normalize(normal)));

  // Combine all factors
  return material.baseColor * lambertFactor * light.color * light.intensity * attenuationFactor;
}

// ============================================================================


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

  // Evaluate radiance from each light
  for(var i = 0u; i < u32(uniforms.nbLights); i++){
    output_color += evaluateRadiance(
      uniforms.lights[i],
      input.worldPos.xyz,
      input.normal.xyz,
      material
    );
  }

  return vec4f(output_color, 1.0);
}
