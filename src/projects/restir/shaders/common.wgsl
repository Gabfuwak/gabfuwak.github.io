// === common.wgsl ===
// Shared structs, bindings, BVH traversal, ray generation, BRDF, sampling

// --- PCG RNG ---
fn pcg_hash(seed: u32) -> u32 {
  var state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn random(t: vec2<f32>) -> f32 {
  let seed = pcg_hash(bitcast<u32>(t.x) ^ pcg_hash(bitcast<u32>(t.y)));
  return f32(seed) / f32(0xFFFFFFFFu);
}

struct PointLight {
  position:  vec3f,
  intensity: f32,
  color:     vec3f,
  _pad1:     f32,
  direction: vec3f,
  angle:     f32,
};

struct Material {
  baseColor: vec3f,
  roughness: f32,
  fresnel:   vec3f,
  metalness: f32,
  emission:  f32,
  _pad1:     f32,
  _pad2:     f32,
  _pad3:     f32,
};

struct Uniforms {
  camera_pos:     vec3f,
  fov_factor:     f32,
  camera_forward: vec3f,
  aspect_ratio:   f32,
  camera_right:   vec3f,
  nb_lights:      f32,
  camera_up:      vec3f,
  frame_count:     f32,
  lights:         array<PointLight, 4>,
  nb_materials:    f32,
  bvh_vis_depth:   f32,
  bvh_heat_max:    f32,
  bvh_early_stop:  f32,
  screen_width:    f32,
  screen_height:   f32,
  render_mode:     f32,
  emissive_count:  f32,
  ris_samples:     f32,
  ris_target:      f32,
  _pad_ris2:       f32,
  _pad_ris3:       f32,
  materials:      array<Material, 16>,
};

struct BVHNode {
  minCorner:          vec3f,
  isLeafOrTriNb:      u32,
  maxCorner:          vec3f,
  triangleOrChildIdx: u32,
};

struct Instance {
  transform:    mat4x4f,
  invTransform: mat4x4f,
  blasOffset:   u32,
  indexOffset:  u32,
  materialId:   u32,
  _pad:         u32,
};

@group(0) @binding(0) var<uniform>       u:         Uniforms;
@group(0) @binding(1) var<storage, read> vertices:  array<f32>;
@group(0) @binding(2) var<storage, read> indices:   array<u32>;
@group(0) @binding(3) var<storage, read> normals:   array<f32>;
@group(0) @binding(4) var<storage, read> uvs:       array<f32>;
@group(0) @binding(5) var<storage, read> tlas:      array<BVHNode>;
@group(0) @binding(6) var<storage, read> blases:    array<BVHNode>;
@group(0) @binding(7) var<storage, read> instances: array<Instance>;
@group(0) @binding(8) var<storage, read_write> accum: array<vec4f>;

struct EmissiveTriangle {
  triIndex:      u32,
  instanceIndex: u32,
  area:          f32,
  cdf:           f32,
};
@group(0) @binding(9) var<storage, read> emissive_tris: array<EmissiveTriangle>;
@group(0) @binding(10) var<storage, read_write> reservoir: array<vec4f>;

// ============================================================================
// BRDF
// ============================================================================

fn GGX_distribution(material: Material, halfVector: vec3f, normal: vec3f) -> f32 {
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

fn schlick_fresnel(material: Material, viewDir: vec3f, halfVector: vec3f) -> vec3f {
  let f_0 = mix(material.fresnel, material.baseColor, material.metalness);
  let cosTheta = max(0.0, dot(viewDir, halfVector));
  return f_0 + (1.0 - f_0) * pow(max(0.0, 1.0 - cosTheta), 5.0);
}

fn G_1_schlick_approx(v: vec3f, normal: vec3f, k: f32) -> f32 {
  let NdotV = max(0.0, dot(normal, v));
  let denom = NdotV * (1.0 - k) + k;
  return NdotV / max(0.0001, denom);
}

fn smith_geometric(material: Material, normal: vec3f, viewDir: vec3f, lightDir: vec3f) -> f32 {
  let roughness = max(0.045, material.roughness);
  let alpha = roughness * roughness;
  let k = alpha * sqrt(2.0 / 3.14159265359);
  return G_1_schlick_approx(viewDir, normal, k) * G_1_schlick_approx(lightDir, normal, k);
}

fn microfacet_BRDF(material: Material, normal: vec3f, viewDir: vec3f, lightDir: vec3f, halfVector: vec3f) -> vec3f {
  let d = GGX_distribution(material, halfVector, normal);
  let f = schlick_fresnel(material, viewDir, halfVector);
  let g = smith_geometric(material, normal, viewDir, lightDir);

  let numerator = d * f * g;
  let NdotL = max(0.0, dot(normal, lightDir));
  let NdotV = max(0.0, dot(normal, viewDir));
  let denominator = max(0.0001, 4.0 * NdotL * NdotV);

  return numerator / denominator;
}

fn evaluateBRDF(mat: Material, normal: vec3f, viewDir: vec3f, lightDir: vec3f) -> vec3f {
  let halfVec = normalize(viewDir + lightDir);
  let F = schlick_fresnel(mat, viewDir, halfVec);
  let kD = (1.0 - mat.metalness) * (vec3f(1.0) - F);
  let f_spec = microfacet_BRDF(mat, normal, viewDir, lightDir, halfVec);
  return kD * mat.baseColor / 3.14159265359 + f_spec;
}

fn evaluateRadiance(
  light: PointLight,
  worldPos: vec3f,
  normal_in: vec3f,
  viewDir: vec3f,
  material: Material
) -> vec3f {
  let normal = normalize(normal_in);
  let lightVec = light.position - worldPos;
  let lightDir = normalize(lightVec);
  let lightToPoint = -lightDir;
  let inCone = dot(normalize(light.direction), lightToPoint) >= cos(light.angle / 2.0);
  if (!inCone) { return vec3f(0.0); }

  let lightDistance = length(lightVec) / 1000.0;
  let attenuationFactor = 1.0 / max(lightDistance * lightDistance, 0.0001);
  let lambertFactor = max(0.0, dot(lightDir, normal));

  return evaluateBRDF(material, normal, viewDir, lightDir)
       * lambertFactor * light.color * light.intensity * attenuationFactor;
}

// ============================================================================
// Ray tracing
// ============================================================================

struct Ray {
  origin:    vec3f,
  direction: vec3f,
};

struct TriHit {
  barycentricCoords: vec3f,
  t:                 f32,
};

struct Hit {
  instanceIndex:     u32,
  triIndex:          u32,
  barycentricCoords: vec3f,
  t:                 f32,
};

struct SurfacePoint {
  world_pos: vec3f,
  normal:    vec3f,
  material:  Material,
};

fn ray_at(screen_coord: vec2f) -> Ray {
  let horizontal = screen_coord.x * u.fov_factor * u.aspect_ratio;
  let vertical   = screen_coord.y * u.fov_factor;
  var ray: Ray;
  ray.origin    = u.camera_pos;
  ray.direction = normalize(
    u.camera_forward +
    horizontal * u.camera_right +
    vertical   * u.camera_up
  );
  return ray;
}

fn rayBoxHit(ray: Ray, box: BVHNode) -> f32 {
  var tmin = 0.0f;
  var tmax = 1e30;
  for (var i = 0; i < 3; i++) {
    let t1 = (box.minCorner[i] - ray.origin[i]) / ray.direction[i];
    let t2 = (box.maxCorner[i] - ray.origin[i]) / ray.direction[i];
    tmin = max(tmin, min(t1, t2));
    tmax = min(tmax, max(t1, t2));
  }
  if (tmax >= tmin && tmax >= 0.0f) { return tmin; }
  else { return 1e30; }
}

fn rayTriangleHit(ray: Ray, tri_idx: u32) -> TriHit {
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

  var no_hit: TriHit;
  no_hit.t = -1.0;
  if (det < 0.00001) { return no_hit; }

  let inv_det = 1.0 / det;
  let T = ray.origin - v0;
  let u_coord = dot(T, P) * inv_det;
  if (u_coord < 0.0 || u_coord > 1.0) { return no_hit; }

  let Q = cross(T, e1);
  let v_coord = dot(ray.direction, Q) * inv_det;
  if (v_coord < 0.0 || u_coord + v_coord > 1.0) { return no_hit; }

  let t = dot(e2, Q) * inv_det;
  if (t <= 0.00001) { return no_hit; }

  var hit: TriHit;
  hit.barycentricCoords = vec3(1.0 - u_coord - v_coord, u_coord, v_coord);
  hit.t                 = t;
  return hit;
}

fn traceBLAS(
  lr: Ray, inst_idx: u32, inst: Instance,
  closest_t: ptr<function, f32>, hit: ptr<function, Hit>, is_shadow: bool
) -> bool {
  var found = false;
  var stack: array<u32, 64>;
  var sp: i32 = 1;
  stack[0] = inst.blasOffset;

  while (sp > 0) {
    sp -= 1;
    let bn = stack[sp];
    if (rayBoxHit(lr, blases[bn]) >= *closest_t) { continue; }

    if (blases[bn].isLeafOrTriNb != 0u) {
      for (var i = 0u; i < blases[bn].isLeafOrTriNb; i++) {
        let tri = inst.indexOffset + blases[bn].triangleOrChildIdx + i;
        let candidate = rayTriangleHit(lr, tri);
        if (candidate.t > 0.0 && candidate.t < *closest_t) {
          *closest_t           = candidate.t;
          found                = true;
          (*hit).instanceIndex = inst_idx;
          (*hit).triIndex      = tri;
          (*hit).barycentricCoords = candidate.barycentricCoords;
          (*hit).t             = candidate.t;
          if (is_shadow) { return true; }
        }
      }
    } else {
      let right = inst.blasOffset + blases[bn].triangleOrChildIdx;
      let left  = bn + 1u;
      let r_t = rayBoxHit(lr, blases[right]);
      let l_t = rayBoxHit(lr, blases[left]);
      if (l_t < r_t) {
        if (r_t < *closest_t) { stack[sp] = right; sp += 1; }
        if (l_t < *closest_t) { stack[sp] = left;  sp += 1; }
      } else {
        if (l_t < *closest_t) { stack[sp] = left;  sp += 1; }
        if (r_t < *closest_t) { stack[sp] = right; sp += 1; }
      }
    }
  }

  return found;
}

fn rayTrace(ray: Ray, hit: ptr<function, Hit>, is_shadow: bool, max_t: f32) -> bool {
  var closest_t = max_t;
  var found_hit = false;

  var tlas_stack: array<u32, 32>;
  var tlas_sp: i32 = 1;
  tlas_stack[0] = 0u;

  while (tlas_sp > 0) {
    tlas_sp -= 1;
    let tn = tlas_stack[tlas_sp];
    if (rayBoxHit(ray, tlas[tn]) >= closest_t) { continue; }

    if (tlas[tn].isLeafOrTriNb != 0u) {
      let inst_idx = tlas[tn].triangleOrChildIdx;
      let inst = instances[inst_idx];
      var lr: Ray;
      lr.origin    = (inst.invTransform * vec4f(ray.origin,    1.0)).xyz;
      lr.direction = (inst.invTransform * vec4f(ray.direction, 0.0)).xyz;

      let blas_hit = traceBLAS(lr, inst_idx, inst, &closest_t, hit, is_shadow);
      if (blas_hit) {
        found_hit = true;
        if (is_shadow) { return true; }
      }
    } else {
      let right = tlas[tn].triangleOrChildIdx;
      let left  = tn + 1u;
      let r_t = rayBoxHit(ray, tlas[right]);
      let l_t = rayBoxHit(ray, tlas[left]);
      if (l_t < r_t) {
        if (r_t < closest_t) { tlas_stack[tlas_sp] = right; tlas_sp += 1; }
        if (l_t < closest_t) { tlas_stack[tlas_sp] = left;  tlas_sp += 1; }
      } else {
        if (l_t < closest_t) { tlas_stack[tlas_sp] = left;  tlas_sp += 1; }
        if (r_t < closest_t) { tlas_stack[tlas_sp] = right; tlas_sp += 1; }
      }
    }
  }

  return found_hit;
}

fn resolve_hit(hit: Hit) -> SurfacePoint {
  let inst = instances[hit.instanceIndex];
  let i0 = indices[hit.triIndex * 3u];
  let i1 = indices[hit.triIndex * 3u + 1u];
  let i2 = indices[hit.triIndex * 3u + 2u];
  let b = hit.barycentricCoords;

  let lp0 = vec3f(vertices[i0*3u], vertices[i0*3u+1u], vertices[i0*3u+2u]);
  let lp1 = vec3f(vertices[i1*3u], vertices[i1*3u+1u], vertices[i1*3u+2u]);
  let lp2 = vec3f(vertices[i2*3u], vertices[i2*3u+1u], vertices[i2*3u+2u]);

  let ln0 = vec3f(normals[i0*3u], normals[i0*3u+1u], normals[i0*3u+2u]);
  let ln1 = vec3f(normals[i1*3u], normals[i1*3u+1u], normals[i1*3u+2u]);
  let ln2 = vec3f(normals[i2*3u], normals[i2*3u+1u], normals[i2*3u+2u]);

  let localPos    = b.x * lp0 + b.y * lp1 + b.z * lp2;
  let localNormal = b.x * ln0 + b.y * ln1 + b.z * ln2;

  var sp: SurfacePoint;
  sp.world_pos = (inst.transform * vec4f(localPos, 1.0)).xyz;
  let inv3 = mat3x3f(inst.invTransform[0].xyz, inst.invTransform[1].xyz, inst.invTransform[2].xyz);
  sp.normal    = normalize(transpose(inv3) * localNormal);
  sp.material  = u.materials[inst.materialId];
  return sp;
}

// ============================================================================
// BVH heat-map visualization
// ============================================================================

fn blasHeat(lr: Ray, inst: Instance, target_depth: i32, base_depth: i32, max_t: f32) -> i32 {
  var count = 0;
  var sn: array<u32, 64>;
  var sd: array<i32, 64>;
  sn[0] = inst.blasOffset; sd[0] = base_depth;
  var sp: i32 = 1;

  while (sp > 0) {
    sp -= 1;
    let bn  = sn[sp];
    let dep = sd[sp];

    if (rayBoxHit(lr, blases[bn]) >= max_t) { continue; }
    if (dep == target_depth || blases[bn].isLeafOrTriNb != 0u) {
      count += 1;
      continue;
    }

    let right = inst.blasOffset + blases[bn].triangleOrChildIdx;
    let left  = bn + 1u;
    sn[sp] = right; sd[sp] = dep + 1; sp += 1;
    sn[sp] = left;  sd[sp] = dep + 1; sp += 1;
  }

  return count;
}

fn bvhHeat(ray: Ray, max_t: f32) -> f32 {
  let target_depth = i32(u.bvh_vis_depth);
  var hit_count = 0;

  var tlas_sn: array<u32, 32>;
  var tlas_sd: array<i32, 32>;
  tlas_sn[0] = 0u; tlas_sd[0] = 0;
  var tsp: i32 = 1;

  while (tsp > 0) {
    tsp -= 1;
    let tn  = tlas_sn[tsp];
    let dep = tlas_sd[tsp];

    if (rayBoxHit(ray, tlas[tn]) >= max_t) { continue; }
    if (dep == target_depth) { hit_count += 1; continue; }

    if (tlas[tn].isLeafOrTriNb != 0u) {
      let inst = instances[tlas[tn].triangleOrChildIdx];
      var lr: Ray;
      lr.origin    = (inst.invTransform * vec4f(ray.origin,    1.0)).xyz;
      lr.direction = (inst.invTransform * vec4f(ray.direction, 0.0)).xyz;
      hit_count += blasHeat(lr, inst, target_depth, dep + 1, max_t);
    } else {
      let right = tlas[tn].triangleOrChildIdx;
      let left  = tn + 1u;
      tlas_sn[tsp] = right; tlas_sd[tsp] = dep + 1; tsp += 1;
      tlas_sn[tsp] = left;  tlas_sd[tsp] = dep + 1; tsp += 1;
    }
  }

  let heat = f32(hit_count) / max(u.bvh_heat_max, 1.0);
  if (heat > 1.0) { return -1.0; }
  return heat;
}

// ============================================================================
// BVH heat-map overlay (shared by both shaders)
// ============================================================================

fn applyBvhOverlay(screen_pos: vec2f, color_in: vec3f) -> vec3f {
  var color = color_in;
  if (u.bvh_vis_depth >= 0.0) {
    let primary_ray = ray_at(screen_pos);
    var heat_max_t = 1e30;
    if (u.bvh_early_stop > 0.0) {
      var es_hit: Hit;
      if (rayTrace(primary_ray, &es_hit, false, 1e30)) {
        heat_max_t = es_hit.t + 0.001;
      }
    }
    let heat = bvhHeat(primary_ray, heat_max_t);
    if (heat < 0.0) {
      color = vec3f(1.0);
    } else {
      color = mix(color, vec3f(1.0, 0.0, 0.0), heat);
    }
  }
  return color;
}

// ============================================================================
// Cosine-weighted hemisphere sampling
// ============================================================================

fn sampleSemiSphere(n: vec3f, seed: f32) -> vec3f {
  let s0 = pcg_hash(bitcast<u32>(seed));
  let s1 = pcg_hash(s0);
  let phi       = f32(s0) / f32(0xFFFFFFFFu) * 6.28318530718;
  let cos_theta = sqrt(f32(s1) / f32(0xFFFFFFFFu));
  let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

  let up        = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.9);
  let tangent   = normalize(cross(up, n));
  let bitangent = cross(n, tangent);

  return normalize(sin_theta * cos(phi) * tangent
                 + sin_theta * sin(phi) * bitangent
                 + cos_theta            * n);
}

// ============================================================================
// Sampling PDFs
// ============================================================================

fn cosinePdf(normal: vec3f, wi: vec3f) -> f32 {
  return max(0.0, dot(normal, wi)) / 3.14159265359;
}

fn neePdf(emission: f32, total_flux: f32, dist: f32, cos_light: f32) -> f32 {
  return (emission / total_flux) * dist * dist / max(0.0001, cos_light);
}

fn brdfPdf(mat: Material, normal: vec3f, viewDir: vec3f, wi: vec3f) -> f32 {
  let NdotL = max(0.0, dot(normal, wi));
  if (NdotL <= 0.0) { return 0.0; }
  let halfVec = normalize(viewDir + wi);
  let NdotH = max(0.0, dot(normal, halfVec));
  let VdotH = max(0.0, dot(viewDir, halfVec));
  let D = GGX_distribution(mat, halfVec, normal);
  let p_diffuse = 0.5 * (1.0 - mat.metalness);
  return p_diffuse * NdotL / 3.14159265359 + (1.0 - p_diffuse) * D * NdotH / max(0.0001, 4.0 * VdotH);
}

// ============================================================================
// GGX importance sampling — sample half-vector from GGX NDF
// ============================================================================

fn sampleGGXHalfVector(normal: vec3f, roughness: f32, seed: u32) -> vec3f {
  let r = max(0.045, roughness);
  let alpha = r * r;
  let alpha_sq = alpha * alpha;

  let s1 = pcg_hash(seed);
  let u1 = f32(s1) / f32(0xFFFFFFFFu);
  let u2 = f32(pcg_hash(s1)) / f32(0xFFFFFFFFu);

  let cos_theta = sqrt((1.0 - u1) / (1.0 + (alpha_sq - 1.0) * u1));
  let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  let phi = 2.0 * 3.14159265359 * u2;

  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
  let tangent   = normalize(cross(up, normal));
  let bitangent = cross(normal, tangent);

  return normalize(sin_theta * cos(phi) * tangent
                 + sin_theta * sin(phi) * bitangent
                 + cos_theta            * normal);
}

// ============================================================================
// Emissive triangle sampling (NEE)
// ============================================================================

// Binary search the CDF: returns index of first entry where cdf >= target
fn emissiveCdfSearch(threshold: f32) -> u32 {
  var lo = 0u;
  var hi = u32(u.emissive_count) - 1u;
  while (lo < hi) {
    let mid = (lo + hi) / 2u;
    if (emissive_tris[mid].cdf < threshold) {
      lo = mid + 1u;
    } else {
      hi = mid;
    }
  }
  return lo;
}

struct LightSample {
  position:  vec3f,   // sampled point on light
  normal:    vec3f,   // light surface normal at that point
  emission:  f32,     // emission value of the material
  pdf:       f32,     // probability density of this sample
};

// Sample a random point on an emissive triangle.
// u1, u2: uniform random numbers in [0,1)
// u_light: uniform random in [0,1) for triangle selection
fn sampleEmissiveTriangle(u_light: f32, u1: f32, u2: f32) -> LightSample {
  var sample: LightSample;

  let total_power = emissive_tris[u32(u.emissive_count) - 1u].cdf;
  let threshold = u_light * total_power;
  let idx = emissiveCdfSearch(threshold);
  let tri = emissive_tris[idx];

  // Fetch triangle vertices from unified buffers
  let i0 = indices[tri.triIndex * 3u];
  let i1 = indices[tri.triIndex * 3u + 1u];
  let i2 = indices[tri.triIndex * 3u + 2u];

  let v0 = vec3f(vertices[i0*3u], vertices[i0*3u+1u], vertices[i0*3u+2u]);
  let v1 = vec3f(vertices[i1*3u], vertices[i1*3u+1u], vertices[i1*3u+2u]);
  let v2 = vec3f(vertices[i2*3u], vertices[i2*3u+1u], vertices[i2*3u+2u]);

  // Transform to world space
  let inst = instances[tri.instanceIndex];
  let w0 = (inst.transform * vec4f(v0, 1.0)).xyz;
  let w1 = (inst.transform * vec4f(v1, 1.0)).xyz;
  let w2 = (inst.transform * vec4f(v2, 1.0)).xyz;

  // Uniform random point on triangle (square-root parameterization)
  let sqrt_u1 = sqrt(u1);
  let b0 = 1.0 - sqrt_u1;
  let b1 = u2 * sqrt_u1;
  sample.position = b0 * w0 + b1 * w1 + (1.0 - b0 - b1) * w2;

  // Triangle normal (world space)
  sample.normal = normalize(cross(w1 - w0, w2 - w0));

  // Emission from material
  sample.emission = u.materials[inst.materialId].emission;

  // PDF = 1/area of this triangle, weighted by selection probability
  // P(pick this triangle) = power_i / total_power
  // P(pick this point | triangle) = 1 / area_i
  // Combined: P = (emission_i * area_i) / total_power * (1 / area_i) = emission_i / total_power
  sample.pdf = sample.emission / total_power;

  return sample;
}
