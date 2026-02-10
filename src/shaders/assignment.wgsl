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
  camera_pos: vec3<f32>,
  nbLights: f32,
  lights: array<PointLight, 4>,
  nbMaterials: f32,
  _pad4: f32,
  _pad5: f32,
  _pad6: f32,
  materials: array<Material, 16>,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;


fn GGX_distribution(materialId: u32, halfVector: vec3<f32>, normal:vec3<f32>) -> f32 {
  let roughness = max(0.045, uniforms.materials[materialId].roughness);
  let alpha = roughness * roughness;
  let alpha_sq = alpha * alpha;

  let pi = 3.14159265359;

  let NdotH = max(0.0, dot(normal, halfVector));
  let NdotH_sq = NdotH * NdotH;

  let denom_inner = NdotH_sq * (alpha_sq - 1.0) + 1.0;
  let denom = pi * denom_inner * denom_inner;

  return alpha_sq / max(1e-7, denom);
}

fn schlick_fresnel(materialId: u32, viewDir: vec3<f32>, halfVector: vec3<f32>) -> vec3<f32>{
  let f_0: vec3<f32> = uniforms.materials[materialId].fresnel;
  let cosTheta = max(0.0, dot(viewDir, halfVector));
  return f_0 + (1.0 - f_0) * pow(max(0.0, 1.0 - cosTheta), 5.0);
}

fn G_1_schlick_approx(vec: vec3<f32>, normal: vec3<f32>, k: f32) -> f32{
  let NdotV = max(0.0, dot(normal, vec));
  let denom = NdotV * (1.0 - k) + k;
  return NdotV / max(0.0001, denom);
}

fn smith_geometric(materialId: u32, normal: vec3<f32>, viewDir: vec3<f32>, lightDir: vec3<f32>) -> f32 {
  let roughness = max(0.045, uniforms.materials[materialId].roughness);
  let alpha = roughness * roughness;
  let k = alpha * sqrt(2.0 / 3.14159265359); // GGX Schlick approximation
  return G_1_schlick_approx(viewDir, normal, k) * G_1_schlick_approx(lightDir, normal, k);
}

fn microfacet_BRDF(materialId: u32, normal: vec3<f32>, viewDir: vec3<f32>, lightDir: vec3<f32>, halfVector: vec3<f32>) -> vec3<f32> {
  let d = GGX_distribution(materialId, halfVector, normal);
  let f = schlick_fresnel(materialId, viewDir, halfVector);
  let g = smith_geometric(materialId, normal, viewDir, lightDir);

  let numerator = d * f * g;
  let NdotL = max(0.0, dot(normal, lightDir));
  let NdotV = max(0.0, dot(normal, viewDir));
  let denominator = max(0.0001, 4.0 * NdotL * NdotV);

  return numerator / denominator;
}

// ============================================================================
// Shared shading function - can be copy-pasted between shaders
// ============================================================================

fn evaluateRadiance(
  materialId: u32,
  light: PointLight,
  worldPos: vec3<f32>,
  normal_in: vec3<f32>,
  viewDir: vec3<f32>,
  material: Material
) -> vec3<f32> {
  let normal = normalize(normal_in);
  // lightDir: from surface toward light
  let lightVec = light.position - worldPos;
  let lightDir = normalize(lightVec);
  let halfVector = normalize(viewDir + lightDir);

  // Check if point is within the light's cone
  let lightToPoint = -lightDir; // Direction from light to point
  let coneAngle = light.angle;
  let inCone = dot(normalize(light.direction), lightToPoint) >= cos(coneAngle / 2.0);

  if (!inCone) {
    return vec3<f32>(0.0); // Outside cone, no contribution
  }

  // Attenuation
  let lightDistance = length(lightVec) / 500.0;
  let constant = 1.0;
  let linear = 0.09;
  let quadratic = 0.032;
  let attenuationFactor = 1.0 / (constant + linear * lightDistance + quadratic * lightDistance * lightDistance);

  // Lambert shading
  let lambertFactor = max(0.0, dot(lightDir, normalize(normal)));

  let diffuse_term = material.baseColor * lambertFactor * light.color * light.intensity * attenuationFactor;
  let specular_term = microfacet_BRDF(materialId, normal, viewDir, lightDir, halfVector) * lambertFactor * light.color * light.intensity * attenuationFactor;

  let kD = (1.0 - uniforms.materials[materialId].metalness) * (1.0 - uniforms.materials[materialId].fresnel);

  return kD * diffuse_term + specular_term;
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

  // viewDir: from surface toward camera
  let viewDir = normalize(uniforms.camera_pos - input.worldPos.xyz);

  // Evaluate radiance from each light
  for(var i = 0u; i < u32(uniforms.nbLights); i++){
    output_color += evaluateRadiance(
      input.objectId,
      uniforms.lights[i],
      input.worldPos.xyz,
      input.normal.xyz,
      viewDir,
      material
    );
  }

  return vec4f(output_color, 1.0);
}
