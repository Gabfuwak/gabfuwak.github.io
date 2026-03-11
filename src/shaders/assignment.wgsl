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
  emission: f32,
  basePerlinFreq: f32,
  perlinOctaveAmp: f32,
  perlinOctaveNb: f32,
}

struct BVHNode{
  minCorner: vec3<f32>,
  isLeafOrTriNb: u32,
  maxCorner: vec3<f32>,
  triangleOrChildIdx: u32,
}

struct Uniforms {
  // Rasterizer (indices 0-15)
  mvp: mat4x4<f32>,
  // Shared (indices 16+)
  camera_pos: vec3<f32>,
  nbLights: f32,
  lights: array<PointLight, 4>,
  nbMaterials: f32,
  spp: f32,
  frame_count: f32,
  canvas_width: f32,
  materials: array<Material, 16>,
  // Raytracer (indices 200-211, fits in the existing UNIFORM_LENGTH buffer)
  camera_forward: vec3f,
  fov_factor: f32,
  camera_right: vec3f,
  aspect_ratio: f32,
  camera_up: vec3f,
  time: f32,
};

@group(0) @binding(0) var<uniform>          uniforms  : Uniforms;
@group(0) @binding(1) var<storage, read>    vertices  : array<f32>;
@group(0) @binding(2) var<storage, read>    indices   : array<u32>;
@group(0) @binding(3) var<storage, read>    objectIds : array<u32>;
@group(0) @binding(4) var<storage, read>    normals   : array<f32>;
@group(0) @binding(5) var<storage, read>    uvs       : array<f32>;
@group(0) @binding(6) var<storage, read>    bvh       : array<BVHNode>;
@group(0) @binding(7) var<storage, read_write> accum: array<vec4f>;

// ============================================================================
// Shared BRDF
// ============================================================================

fn GGX_distribution(material: Material, halfVector: vec3<f32>, normal:vec3<f32>) -> f32 {
  let roughness = max(0.045, material.roughness);
  let alpha = roughness * roughness;
  let alpha_sq = alpha * alpha;

  let pi = 3.14159265359;

  let NdotH = max(0.0, dot(normal, halfVector));
  let NdotH_sq = NdotH * NdotH;

  let denom_inner = NdotH_sq * (alpha_sq - 1.0) + 1.0;
  let denom = pi * denom_inner * denom_inner;

  return alpha_sq / max(1e-7, denom);
}

fn schlick_fresnel(material: Material, viewDir: vec3<f32>, halfVector: vec3<f32>) -> vec3<f32>{
  let f_0: vec3<f32> = material.fresnel;
  let cosTheta = max(0.0, dot(viewDir, halfVector));
  return f_0 + (1.0 - f_0) * pow(max(0.0, 1.0 - cosTheta), 5.0);
}

fn G_1_schlick_approx(vec: vec3<f32>, normal: vec3<f32>, k: f32) -> f32{
  let NdotV = max(0.0, dot(normal, vec));
  let denom = NdotV * (1.0 - k) + k;
  return NdotV / max(0.0001, denom);
}

fn smith_geometric(material: Material, normal: vec3<f32>, viewDir: vec3<f32>, lightDir: vec3<f32>) -> f32 {
  let roughness = max(0.045, material.roughness);
  let alpha = roughness * roughness;
  let k = alpha * sqrt(2.0 / 3.14159265359);
  return G_1_schlick_approx(viewDir, normal, k) * G_1_schlick_approx(lightDir, normal, k);
}

fn microfacet_BRDF(material: Material, normal: vec3<f32>, viewDir: vec3<f32>, lightDir: vec3<f32>, halfVector: vec3<f32>) -> vec3<f32> {
  let d = GGX_distribution(material, halfVector, normal);
  let f = schlick_fresnel(material, viewDir, halfVector);
  let g = smith_geometric(material, normal, viewDir, lightDir);

  let numerator = d * f * g;
  let NdotL = max(0.0, dot(normal, lightDir));
  let NdotV = max(0.0, dot(normal, viewDir));
  let denominator = max(0.0001, 4.0 * NdotL * NdotV);

  return numerator / denominator;
}

fn evaluateRadiance(
  light: PointLight,
  worldPos: vec3<f32>,
  normal_in: vec3<f32>,
  viewDir: vec3<f32>,
  material: Material
) -> vec3<f32> {
  let normal = normalize(normal_in);
  let lightVec = light.position - worldPos;
  let lightDir = normalize(lightVec);
  let halfVector = normalize(viewDir + lightDir);

  let lightToPoint = -lightDir;
  let coneAngle = light.angle;
  let inCone = dot(normalize(light.direction), lightToPoint) >= cos(coneAngle / 2.0);

  if (!inCone) {
    return vec3f(material.emission);
  }

  let lightDistance = length(lightVec) / 100.0;
  let constant = 1.0;
  let linear = 0.09;
  let quadratic = 0.032;
  let attenuationFactor = 1.0 / (constant + linear * lightDistance + quadratic * lightDistance * lightDistance);

  let lambertFactor = max(0.0, dot(lightDir, normalize(normal)));

  let diffuse_term = material.baseColor * lambertFactor * light.color * light.intensity * attenuationFactor;
  let specular_term = microfacet_BRDF(material, normal, viewDir, lightDir, halfVector) * lambertFactor * light.color * light.intensity * attenuationFactor;

  let kD = (1.0 - material.metalness) * (1.0 - material.fresnel);

  return kD * diffuse_term + specular_term + material.emission;
}


// ============================================================================
// Rasterizer
// ============================================================================

struct RastVertexOutput {
  @builtin(position) pos : vec4f,
  @location(0) @interpolate(flat) objectId: u32,
  @location(1) normal: vec4f,
  @location(2) worldPos: vec4f,
  @location(3) uv: vec2f,
};

@vertex
fn rastVertexMain(@builtin(vertex_index) vid: u32) -> RastVertexOutput {
  let idx = indices[vid];
  let pos    = vec3f(vertices[idx*3u], vertices[idx*3u+1u], vertices[idx*3u+2u]);
  let normal = vec3f(normals[idx*3u],  normals[idx*3u+1u],  normals[idx*3u+2u]);
  let uv     = vec2f(uvs[idx*2u], uvs[idx*2u+1u]);
  let objId  = objectIds[idx];

  var output: RastVertexOutput;
  output.pos      = uniforms.mvp * vec4f(pos, 1.0);
  output.worldPos = vec4f(pos, 1.0);
  output.objectId = objId;
  output.normal   = vec4f(normal, 0.0);
  output.uv       = uv;
  return output;
}

@fragment
fn rastFragmentMain(input: RastVertexOutput) -> @location(0) vec4f {
  var output_color = vec3f(0.0);

  if (u32(uniforms.nbLights) == 0u) {
    return vec4f(output_color, 1.0);
  }

  let material = resolve_material(input.objectId, input.uv);
  let viewDir  = normalize(uniforms.camera_pos - input.worldPos.xyz);

  for(var i = 0u; i < u32(uniforms.nbLights); i++) {
    output_color += evaluateRadiance(
      uniforms.lights[i],
      input.worldPos.xyz,
      input.normal.xyz,
      viewDir,
      material
    );
  }

  return vec4f(output_color, 1.0);
}


// ============================================================================
// Raytracer
// ============================================================================

struct RayVertexOutput {
  @builtin(position) clip_pos: vec4f,
  @location(0) screen_pos: vec2f,
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

struct SurfacePoint {
  world_pos: vec3f,
  normal:    vec3f,
  uv:    vec2f,
  objectId:  u32,
  material:  Material,
};

fn resolve_material(material_id: u32, uv: vec2f) -> Material {
  var mat = uniforms.materials[material_id];
  switch(material_id) { // switch stays so that later i can add a custom material scheme, maybe with graphs we'll see
    default: {
      if(mat.perlinOctaveAmp > 1e-30){
        let t = uniforms.time / 100.0;
        mat.baseColor += vec3f(octavePerlin3D(vec3f(uv, t), mat.basePerlinFreq, mat.perlinOctaveAmp, u32(mat.perlinOctaveNb)));
        mat.roughness += octavePerlin3D(vec3f(uv + 1.0, t), mat.basePerlinFreq, mat.perlinOctaveAmp, u32(mat.perlinOctaveNb));
        mat.metalness += octavePerlin3D(vec3f(uv + 2.0, t), mat.basePerlinFreq, mat.perlinOctaveAmp, u32(mat.perlinOctaveNb));
      }
      return mat;
    }
  }
}

fn resolve_hit(hit: Hit) -> SurfacePoint {
  let i0 = indices[hit.triIndex * 3u];
  let i1 = indices[hit.triIndex * 3u + 1u];
  let i2 = indices[hit.triIndex * 3u + 2u];

  let bary = hit.barycentricCoords;

  let p0 = vec3f(vertices[i0*3u], vertices[i0*3u+1u], vertices[i0*3u+2u]);
  let p1 = vec3f(vertices[i1*3u], vertices[i1*3u+1u], vertices[i1*3u+2u]);
  let p2 = vec3f(vertices[i2*3u], vertices[i2*3u+1u], vertices[i2*3u+2u]);

  let n0 = vec3f(normals[i0*3u], normals[i0*3u+1u], normals[i0*3u+2u]);
  let n1 = vec3f(normals[i1*3u], normals[i1*3u+1u], normals[i1*3u+2u]);
  let n2 = vec3f(normals[i2*3u], normals[i2*3u+1u], normals[i2*3u+2u]);


  let uv0 = vec2f(uvs[i0*2u], uvs[i0*2u+1u]);
  let uv1 = vec2f(uvs[i1*2u], uvs[i1*2u+1u]);
  let uv2 = vec2f(uvs[i2*2u], uvs[i2*2u+1u]);

  var sp: SurfacePoint;
  sp.world_pos = bary.x * p0 + bary.y * p1 + bary.z * p2;
  sp.normal    = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
  sp.uv = bary.x * uv0 + bary.y * uv1 + bary.z * uv2;
  sp.objectId  = objectIds[i0];
  sp.material  = resolve_material(sp.objectId, sp.uv);
  return sp;
}

fn ray_at(screenCoord: vec2<f32>) -> Ray {
  let horizontal = screenCoord.x * uniforms.fov_factor * uniforms.aspect_ratio;
  let vertical   = screenCoord.y * uniforms.fov_factor;

  var output: Ray;
  output.origin    = uniforms.camera_pos;
  output.direction = normalize(
    uniforms.camera_forward +
    horizontal * uniforms.camera_right +
    vertical   * uniforms.camera_up
  );
  return output;
}

fn rayBoxHit(ray: Ray, box: BVHNode) -> f32 {
  var tmin = 0.0f;
  var tmax = 1e30;

  for(var i = 0; i < 3; i++){
    let t1 = (box.minCorner[i] - ray.origin[i]) / ray.direction[i];
    let t2 = (box.maxCorner[i] - ray.origin[i]) / ray.direction[i];

    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));
  }


  if(tmax >= tmin && tmax >= 0.0f){
    return tmin;
  }else {
    return 1e30;
  }
}

fn rayTriangleHit(ray: Ray, tri_idx: u32) -> Hit {
  let i0 = indices[tri_idx * 3u];
  let i1 = indices[tri_idx * 3u + 1u];
  let i2 = indices[tri_idx * 3u + 2u];

  let v0 = vec3f(vertices[i0*3u], vertices[i0*3u+1u], vertices[i0*3u+2u]);
  let v1 = vec3f(vertices[i1*3u], vertices[i1*3u+1u], vertices[i1*3u+2u]);
  let v2 = vec3f(vertices[i2*3u], vertices[i2*3u+1u], vertices[i2*3u+2u]);

  let e1 = v1 - v0;
  let e2 = v2 - v0;

  let P   = cross(ray.direction, e2);
  let det = dot(e1, P);

  var no_hit: Hit;
  no_hit.t = -1.0;

  if (det < 0.00001) { return no_hit; }

  let inv_det = 1.0 / det;
  let T = ray.origin - v0;
  let u = dot(T, P) * inv_det;

  if (u < 0.0 || u > 1.0) { return no_hit; }

  let Q = cross(T, e1);
  let v = dot(ray.direction, Q) * inv_det;

  if (v < 0.0 || u + v > 1.0) { return no_hit; }

  let t = dot(e2, Q) * inv_det;

  if (t <= 0.00001) { return no_hit; }

  var hit: Hit;
  hit.triIndex = tri_idx;
  hit.barycentricCoords = vec3(1.0 - u - v, u, v);
  hit.t = t;
  return hit;
}

fn rayTrace(ray: Ray, hit: ptr<function, Hit>, is_shadow: bool, max_t: f32) -> bool {
  var closest_t = max_t;
  var found_hit = false;

  var stack: array<u32, 64>;
  var stack_ptr: i32 = 0;

  stack[0] = 0u;
  stack_ptr = 1;

  while (stack_ptr > 0) {
    stack_ptr -= 1;
    let curr_node_idx = stack[stack_ptr];

    if (bvh[curr_node_idx].isLeafOrTriNb != 0u) {
      for(var candidate_idx = 0u; candidate_idx < bvh[curr_node_idx].isLeafOrTriNb; candidate_idx++){
        let candidate = rayTriangleHit(ray, bvh[curr_node_idx].triangleOrChildIdx + candidate_idx);

        if (candidate.t > 0.0 && candidate.t < closest_t) {
          closest_t = candidate.t;
          found_hit = true;
          *hit = candidate;
          if(is_shadow){ // early stop if anything is blocking a shadow ray
            return true;
          }
        }
      }
    } else {
      let rightChild = bvh[curr_node_idx].triangleOrChildIdx;
      let leftChild  = curr_node_idx + 1;

      let rightChildHit_t = rayBoxHit(ray, bvh[rightChild]);
      let leftChildHit_t  = rayBoxHit(ray, bvh[leftChild]);

      if (leftChildHit_t < rightChildHit_t) {
        if (rightChildHit_t < closest_t) {
          stack[stack_ptr] = rightChild;
          stack_ptr += 1;
        }
        if (leftChildHit_t < closest_t) {
          stack[stack_ptr] = leftChild;
          stack_ptr += 1;
        }
      } else {
        if (leftChildHit_t < closest_t) {
          stack[stack_ptr] = leftChild;
          stack_ptr += 1;
        }
        if (rightChildHit_t < closest_t) {
          stack[stack_ptr] = rightChild;
          stack_ptr += 1;
        }
      }
    }
  }

  return found_hit;
}

@vertex
fn rayVertexMain(@builtin(vertex_index) vid: u32) -> RayVertexOutput {
  // Fullscreen quad, no vertex buffer needed
  let positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0),
  );
  let pos = positions[vid];
  var output: RayVertexOutput;
  output.clip_pos   = vec4f(pos, 0.0, 1.0);
  output.screen_pos = pos;
  return output;
}

// ============================================================================
// Picking
// ============================================================================

@group(0) @binding(8) var<uniform>             pick_coords : vec2f;
@group(0) @binding(9) var<storage, read_write> pick_result : i32;

@compute @workgroup_size(1)
fn pick_main() {
  let ray = ray_at(pick_coords);
  var hit: Hit;
  if (rayTrace(ray, &hit, false, 1e30)) {
    pick_result = i32(objectIds[indices[hit.triIndex * 3u]]);
  } else {
    pick_result = -1;
  }
}

fn sampleSemiSphere(n : vec3f, seed: f32) -> vec3f{
  // Chain PCG in integer space to avoid correlated bitcast of adjacent floats
  let s0 = pcg_hash(bitcast<u32>(seed));
  let s1 = pcg_hash(s0);
  let phi       = f32(s0) / f32(0xFFFFFFFFu) * 6.28318530718;
  let cos_theta = f32(s1) / f32(0xFFFFFFFFu); // [0, 1] for hemisphere
  let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  // add semi sphere variation around the normal

  let up        = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let tangent   = normalize(cross(up, n));
  let bitangent = cross(n, tangent);

  return normalize(sin_theta * cos(phi) * tangent
                 + sin_theta * sin(phi) * bitangent
                 + cos_theta            * n);
}

@fragment
fn rayFragmentMain(input: RayVertexOutput) -> @location(0) vec4f {
  var output_color = vec3f(0.0);

  for(var sample = 0u; sample < u32(uniforms.spp); sample++){
    var ray = ray_at(input.screen_pos);
    var sample_output_color = vec3f(0.0);
    var throughput = vec3f(1.0);
    for (var bounce = 1u; bounce <= 5; bounce++) {
      var hit_data: Hit;
      if (!rayTrace(ray, &hit_data, false, 1e30)) { break; }

      let sp = resolve_hit(hit_data);
      let viewDir = -ray.direction;

      sample_output_color += throughput * sp.material.emission;

      for (var i = 0u; i < u32(uniforms.nbLights); i++) {
        let lightVec = uniforms.lights[i].position - sp.world_pos;
        let lightDir = normalize(lightVec);

        var shadow_ray: Ray;
        shadow_ray.direction = lightDir;
        shadow_ray.origin = sp.world_pos + sp.normal * 0.1;

        var shadow_hit: Hit;
        let is_shadow = rayTrace(shadow_ray, &shadow_hit, true, length(lightVec));

        if (!is_shadow) {
          output_color += throughput * evaluateRadiance(
            uniforms.lights[i],
            sp.world_pos,
            sp.normal,
            viewDir,
            sp.material
          );
        }
      }

      let F = schlick_fresnel(sp.material, viewDir, sp.normal);
      //if (all(throughput * F < vec3f(0.001))) { break; }

      // white reflects more than black so we need a kd here too
      let kD = (1.0 - sp.material.metalness) * (vec3f(1.0) - F);
      throughput *= kD * sp.material.baseColor + F;

      let px = vec2u(input.clip_pos.xy);
      let seed_u = pcg_hash(px.x ^ pcg_hash(px.y ^ pcg_hash(sample ^ pcg_hash(bounce ^ u32(uniforms.frame_count)))));
      let seed = bitcast<f32>(seed_u);
      let random_ray_dir = sampleSemiSphere(sp.normal, seed);

      ray = Ray(sp.world_pos + sp.normal * 0.001, random_ray_dir);
      let rr_seed = pcg_hash(seed_u ^ 0xDEADBEEFu);
      if(all(throughput < vec3f(f32(rr_seed) / f32(0xFFFFFFFFu)))) {break;} // russian roulette
    }
    output_color += sample_output_color;
  }

  let coords = vec2u(input.clip_pos.xy);
  let idx = coords.y * u32(uniforms.canvas_width) + coords.x;
  let sample = output_color / uniforms.spp;
  if (uniforms.frame_count <= 1.0) {
    accum[idx] = vec4f(sample, 1.0);
  } else {
    accum[idx] += vec4f(sample, 0.0);
  }
  return vec4f(accum[idx].rgb / uniforms.frame_count, 1.0);
}
