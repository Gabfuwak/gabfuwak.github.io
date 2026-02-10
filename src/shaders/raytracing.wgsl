struct VertexOutput {
  @builtin(position) clip_pos: vec4f,
  @location(0) screen_pos: vec2f,
};

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

struct RayUniforms {
  ////// RAY TRACER ONLY
  camera_forward: vec3f,
  fov_factor: f32,
  camera_right: vec3f,
  aspect_ratio: f32,
  camera_up: vec3f,
  _pad0: f32,
  _pad1: vec4f,               // unused (indices 12-15)
  ////// SHARED, starting at idx 16
  camera_pos: vec3f,
  nbLights: f32,
  lights: array<PointLight, 4>,
  nbMaterials: f32,
  _pad5: f32,
  _pad6: f32,
  _pad7: f32,
  materials: array<Material, 16>,
};


struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>
};

struct Hit {
  triIndex: u32,
  barycentricCoords: vec3<f32>,
  t: f32
};


@group(0) @binding(0)
var<uniform> uniforms : RayUniforms;

@group(0) @binding(1)
var<storage, read> vertices : array<f32>;  // positions

@group(0) @binding(2)
var<storage, read> indices : array<u32>;     // triangle indices

@group(0) @binding(3)
var<storage, read> objectIds : array<u32>;  // object IDs per vertex

@group(0) @binding(4)
var<storage, read> normals : array<f32>;  // normals


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

fn evaluateRadiance(
  materialId: u32,
  light: PointLight,
  worldPos: vec3<f32>,
  normal: vec3<f32>,
  viewDir: vec3<f32>,
  material: Material,
) -> vec3<f32> {
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


fn rayTrace(ray: Ray, hit: ptr<function, Hit>) -> bool {
  var closest_t = 1e30;
  var found_hit = false;

  let num_triangles = arrayLength(&indices) / 3;
  for (var tri_idx = 0u; tri_idx < num_triangles; tri_idx++) {
    let i0 = indices[tri_idx * 3 + 0];
    let i1 = indices[tri_idx * 3 + 1];
    let i2 = indices[tri_idx * 3 + 2];

    let v0 = vec3f(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    let v1 = vec3f(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    let v2 = vec3f(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

    let e1 = v1 - v0;
    let e2 = v2 - v0;

    let P = cross(ray.direction, e2);
    let det = dot(e1, P);

    if (abs(det) < 0.00001) {
      continue;
    }

    let inv_det = 1.0 / det;
    let T = ray.origin - v0;
    let u = dot(T, P) * inv_det;

    if (u < 0.0 || u > 1.0) {
      continue;
    }

    let Q = cross(T, e1);
    let v = dot(ray.direction, Q) * inv_det;

    if (v < 0.0 || u + v > 1.0) {
      continue;
    }
    
    let t = dot(e2, Q) * inv_det;
    
    if (t > 0.00001 && t < closest_t) {
      closest_t = t;
      found_hit = true;
      (*hit).triIndex = tri_idx;
      (*hit).barycentricCoords = vec3(1.0 - u - v, u, v);
      (*hit).t = closest_t;
    }
  }
  
  return found_hit;
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
  var curr_ray: Ray = ray_at(input.screen_pos);
  var hit_data: Hit;
  var output_color = vec3f(0.0);
  if(rayTrace(curr_ray, &hit_data)){

    let i0 = indices[hit_data.triIndex * 3 + 0];
    let i1 = indices[hit_data.triIndex * 3 + 1];
    let i2 = indices[hit_data.triIndex * 3 + 2];

    // Get objectId (should be same for all vertices of a triangle)
    let objectId = objectIds[i0];
    let material = uniforms.materials[objectId];

    let n0 = vec3f(normals[i0 * 3 + 0], normals[i0 * 3 + 1], normals[i0 * 3 + 2]);
    let n1 = vec3f(normals[i1 * 3 + 0], normals[i1 * 3 + 1], normals[i1 * 3 + 2]);
    let n2 = vec3f(normals[i2 * 3 + 0], normals[i2 * 3 + 1], normals[i2 * 3 + 2]);

    let p0 = vec3f(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    let p1 = vec3f(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    let p2 = vec3f(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);


    let bary = hit_data.barycentricCoords;
    let normal = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
    let world_pos = bary.x * p0 + bary.y * p1 + bary.z * p2;

    let viewDir = normalize(uniforms.camera_pos - world_pos);

    // Evaluate radiance from each light
    for(var i = 0u; i < u32(uniforms.nbLights); i++){
      // Cast shadow ray
      let lightVec = uniforms.lights[i].position - world_pos;
      let lightDir = normalize(lightVec);

      var shadow_ray: Ray;
      shadow_ray.direction = lightDir;
      shadow_ray.origin = world_pos + normal * 0.1;

      var shadow_hit: Hit;
      let is_shadow = rayTrace(shadow_ray, &shadow_hit);

      // Only add light contribution if not in shadow
      if(!is_shadow || shadow_hit.t > length(lightVec)){
        output_color += evaluateRadiance(
          objectId,
          uniforms.lights[i],
          world_pos,
          normal,
          viewDir,
          material
        );
      }
    }
  }

  return vec4f(output_color, 1.0);
}
