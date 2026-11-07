import"./modulepreload-polyfill-B5Qt9EMX.js";const Ee=`// === common.wgsl ===
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

fn microfacet_BRDF(material: Material, normal: vec3f, viewDir: vec3f, lightDir: vec3f, halfVector: vec3f, f: vec3f) -> vec3f {
  let d = GGX_distribution(material, halfVector, normal);
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
  let f_spec = microfacet_BRDF(mat, normal, viewDir, lightDir, halfVec, F);
  return kD * mat.baseColor / 3.14159265359 + f_spec;
}

fn evaluateRadiance(
  light: PointLight,
  worldPos: vec3f,
  normal: vec3f,
  viewDir: vec3f,
  material: Material
) -> vec3f {
  let lightVec   = light.position - worldPos;
  let light_dist = length(lightVec);
  let lightDir   = lightVec / light_dist;
  let lightToPoint = -lightDir;
  let inCone = dot(light.direction, lightToPoint) >= cos(light.angle / 2.0);
  if (!inCone) { return vec3f(0.0); }

  let lightDistance = light_dist / 1000.0;
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

fn rayBoxHit(ray: Ray, inv_dir: vec3f, box: BVHNode) -> f32 {
  var tmin = 0.0f;
  var tmax = 1e30;
  for (var i = 0; i < 3; i++) {
    let t1 = (box.minCorner[i] - ray.origin[i]) * inv_dir[i];
    let t2 = (box.maxCorner[i] - ray.origin[i]) * inv_dir[i];
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
  let inv_dir = 1.0 / lr.direction;

  while (sp > 0) {
    sp -= 1;
    let bn = stack[sp];
    if (rayBoxHit(lr, inv_dir, blases[bn]) >= *closest_t) { continue; }

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
      let r_t = rayBoxHit(lr, inv_dir, blases[right]);
      let l_t = rayBoxHit(lr, inv_dir, blases[left]);
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
  let inv_dir = 1.0 / ray.direction;

  var tlas_stack: array<u32, 16>;
  var tlas_sp: i32 = 1;
  tlas_stack[0] = 0u;

  while (tlas_sp > 0) {
    tlas_sp -= 1;
    let tn = tlas_stack[tlas_sp];
    if (rayBoxHit(ray, inv_dir, tlas[tn]) >= closest_t) { continue; }

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
      let r_t = rayBoxHit(ray, inv_dir, tlas[right]);
      let l_t = rayBoxHit(ray, inv_dir, tlas[left]);
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
  var sn: array<u32, 32>;
  var sd: array<i32, 32>;
  sn[0] = inst.blasOffset; sd[0] = base_depth;
  var sp: i32 = 1;
  let inv_dir = 1.0 / lr.direction;

  while (sp > 0) {
    sp -= 1;
    let bn  = sn[sp];
    let dep = sd[sp];

    if (rayBoxHit(lr, inv_dir, blases[bn]) >= max_t) { continue; }
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
  let inv_dir = 1.0 / ray.direction;

  var tlas_sn: array<u32, 32>;
  var tlas_sd: array<i32, 32>;
  tlas_sn[0] = 0u; tlas_sd[0] = 0;
  var tsp: i32 = 1;

  while (tsp > 0) {
    tsp -= 1;
    let tn  = tlas_sn[tsp];
    let dep = tlas_sd[tsp];

    if (rayBoxHit(ray, inv_dir, tlas[tn]) >= max_t) { continue; }
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

  return sin_theta * cos(phi) * tangent
       + sin_theta * sin(phi) * bitangent
       + cos_theta            * n;
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

  return sin_theta * cos(phi) * tangent
       + sin_theta * sin(phi) * bitangent
       + cos_theta            * normal;
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
`,Ze=`// === raytracer.wgsl ===
// Direct lighting only — single sample per pixel, no bounces.

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.xy;
  let screen_pos = vec2f(f32(px.x) / (u.screen_width * 0.5) - 1.0, 1.0 - f32(px.y) / (u.screen_height * 0.5));
  let ray = ray_at(screen_pos);

  var color = vec3f(0.0);

  var hit: Hit;
  if (!rayTrace(ray, &hit, false, 1e30)) {
    let t = 0.5 * (ray.direction.y + 1.0);
    color = mix(vec3f(1.0), vec3f(0.5, 0.7, 1.0), t);
  } else {
    let sp = resolve_hit(hit);
    let viewDir = -ray.direction;

    // Emission
    color += sp.material.emission;

    // Ambient
    color += 0.3 * sp.material.baseColor;

    // Direct lighting with shadow rays
    for (var i = 0u; i < u32(u.nb_lights); i++) {
      let lightVec   = u.lights[i].position - sp.world_pos;
      let light_dist = length(lightVec);
      let lightDir   = lightVec / light_dist;

      var shadow_ray: Ray;
      shadow_ray.origin    = sp.world_pos + sp.normal * 0.1;
      shadow_ray.direction = lightDir;

      var shadow_hit: Hit;
      if (!rayTrace(shadow_ray, &shadow_hit, true, light_dist)) {
        color += evaluateRadiance(u.lights[i], sp.world_pos, sp.normal, viewDir, sp.material);
      }
    }
  }

  // Touch unused bindings so auto layout keeps them alive
  let _uv = uvs[0];
  let _et = emissive_tris[0];
  let _rv = reservoir[0];

  color = applyBvhOverlay(screen_pos, color);

  let idx = px.y * u32(u.screen_width) + px.x;
  accum[idx] = vec4f(color, 1.0);
}
`,Ke=`// === pathtracer.wgsl ===
// Path tracing with bounce loop, cosine-weighted hemisphere sampling, Russian roulette.

fn neePointLight(sp: SurfacePoint, viewDir: vec3f,
                 u_draw: f32, point_flux: array<f32, 4>, n_point: u32,
                 total_flux: f32) -> vec3f {
  var chosen = n_point - 1u;
  var cum    = 0.0;
  // We calculate flux at sample time for simplicity because I don't plan on having a lot of light sources
  // The "right" way to do it is to precalculate it on the CPU at scene creation just like the emissive lights
  for (var i = 0u; i < n_point; i++) {
    cum += point_flux[i];
    if (u_draw < cum) { chosen = i; break; }
  }
  let light      = u.lights[chosen];
  let lightVec   = light.position - sp.world_pos;
  let light_dist = length(lightVec);
  let lightDir   = lightVec / light_dist;
  var shadow_ray: Ray;
  shadow_ray.origin    = sp.world_pos + sp.normal * 0.001;
  shadow_ray.direction = lightDir;
  var shadow_hit: Hit;
  if (!rayTrace(shadow_ray, &shadow_hit, true, light_dist)) {
    return evaluateRadiance(light, sp.world_pos, sp.normal, viewDir, sp.material)
           * total_flux / point_flux[chosen];
  }
  return vec3f(0.0);
}

fn neeEmissiveTriangle(sp: SurfacePoint, viewDir: vec3f,
                       u_emissive: f32, u1: f32, u2: f32,
                       total_flux: f32, use_mis: bool) -> vec3f {
  let ls          = sampleEmissiveTriangle(u_emissive, u1, u2);
  let lightVec    = ls.position - sp.world_pos;
  let light_dist  = length(lightVec);
  let lightDir    = lightVec/light_dist;
  let dist_m      = light_dist / 1000.0; // mm -> m
  let cos_light   = max(0.0, dot(ls.normal, -lightDir));
  let cos_surface = max(0.0, dot(sp.normal, lightDir));
  if (cos_light <= 0.0 || cos_surface <= 0.0) { return vec3f(0.0); }

  var shadow_ray: Ray;
  shadow_ray.origin    = sp.world_pos + sp.normal * 0.001;
  shadow_ray.direction = lightDir;
  var shadow_hit: Hit;
  if (!rayTrace(shadow_ray, &shadow_hit, true, light_dist - 0.5)) {
    let brdf      = evaluateBRDF(sp.material, sp.normal, viewDir, lightDir);
    let pdf_light = neePdf(ls.emission, total_flux, dist_m, cos_light);
    var w = 1.0;
    if (use_mis) {
      let pdf_brdf = brdfPdf(sp.material, sp.normal, viewDir, lightDir);
      w = pdf_light / (pdf_light + pdf_brdf);
    }
    return brdf * cos_surface * ls.emission / pdf_light * w;
  }
  return vec3f(0.0);
}

@compute @workgroup_size(8, 8)
fn computeMain(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.xy;
  let screen_pos = vec2f(f32(px.x) / (u.screen_width * 0.5) - 1.0, 1.0 - f32(px.y) / (u.screen_height * 0.5));
  var ray = ray_at(screen_pos);

  var color = vec3f(0.0);
  var throughput = vec3f(1.0);
  var prev_pdf = 0.0;

  let use_nee    = (u32(u.render_mode) & 1u) != 0u;
  let use_brdfis = (u32(u.render_mode) & 2u) != 0u;
  let use_mis    = (u32(u.render_mode) & 4u) != 0u;


  // Precompute flux totals (needed by both MIS emission weighting and NEE sampling)
  let n_emissive = u32(u.emissive_count);
  let n_point    = u32(u.nb_lights);
  var point_flux: array<f32, 4>;
  var point_flux_total = 0.0;
  for (var i = 0u; i < n_point; i++) {
    point_flux[i]     = 4.0 * 3.14159265359 * u.lights[i].intensity;
    point_flux_total += point_flux[i];
  }
  let emissive_flux_total = select(0.0, emissive_tris[n_emissive - 1u].cdf, n_emissive > 0u);
  let total_flux = point_flux_total + emissive_flux_total;


  for (var bounce = 0u; bounce <= 32u; bounce++) { // cap at 32 in case we have a degenerate case, RR should rarely make it bounce more than 3-5 times
    var hit: Hit;
    if (!rayTrace(ray, &hit, false, 1e30)) {
      // Sky contribution
      let t = 0.5 * (ray.direction.y + 1.0);
      color += throughput * mix(vec3f(1.0), vec3f(0.5, 0.7, 1.0), t);
      break;
    }

    let sp = resolve_hit(hit);
    let viewDir = -ray.direction;



    // Naive: add emission every bounce. NEE: only bounce 0 (NEE handles the rest).
    // MIS: weight BRDF-sampled emissive hits by balance heuristic on bounce > 0.
    if (!use_nee || bounce == 0u) {
      color += throughput * sp.material.emission;
    } else if (use_mis && sp.material.emission > 0.0 && prev_pdf > 0.0) {
      let cos_light = max(0.0, dot(sp.normal, -ray.direction));
      let dist_m    = hit.t / 1000.0;
      let pdf_light = neePdf(sp.material.emission, total_flux, dist_m, cos_light);
      let w_brdf    = prev_pdf / (prev_pdf + pdf_light);
      color += throughput * sp.material.emission * w_brdf;
    }

    // NEE: point lights (direct) + emissive triangles (RIS reservoir when enabled).
    if (use_nee && total_flux > 0.0) {
      var nee_seed = pcg_hash(px.x ^ pcg_hash(px.y ^ pcg_hash(bounce ^ u32(u.frame_count) ^ 0xABCDEF01u)));

      // Point lights: always direct sample (outside reservoir)
      if (n_point > 0u && point_flux_total > 0.0) {
        let u_draw = f32(nee_seed) / f32(0xFFFFFFFFu) * point_flux_total;
        color += throughput * neePointLight(sp, viewDir, u_draw, point_flux, n_point, point_flux_total);
      }

      // Emissive triangles
      if (n_emissive > 0u && emissive_flux_total > 0.0) {
        let M = u32(u.ris_samples);
        let use_restir = (u32(u.render_mode) & 16u) != 0u;
        let pix_idx = px.y * u32(u.screen_width) + px.x;

        // Load previous reservoir for temporal reuse (ReSTIR)
        var prev_res = vec4f(0.0);
        if (use_restir && bounce == 0u) {
          prev_res = reservoir[pix_idx];
        }
        var res_u_draw = prev_res.x;
        var res_w_sum = prev_res.y;
        var res_target_p = prev_res.z;

        for (var m = 0u; m < M; m++) {
          nee_seed = pcg_hash(nee_seed);
          let cand_u_draw = f32(nee_seed) / f32(0xFFFFFFFFu);

          let ls = sampleEmissiveTriangle(cand_u_draw, 0.5, 0.5); // centroid, not really principled but it's a decent estimation and makes restir easier because less data to carry over
          let lightVec = ls.position - sp.world_pos;
          let light_dist = length(lightVec);
          let lightDir = lightVec / light_dist;
          let dist_m = light_dist / 1000.0;
          let cos_light = max(0.0, dot(ls.normal, -lightDir));
          let cos_surface = max(0.0, dot(sp.normal, lightDir));

          if (cos_light > 0.0 && cos_surface > 0.0) {
            let q = neePdf(ls.emission, total_flux, dist_m, cos_light);
            var p_hat = 0.0;
            let ris_mode = u32(u.ris_target);

            if (ris_mode == 0u) {
              // Full: BRDF + shadow test (p̂ = f)
              let brdf = evaluateBRDF(sp.material, sp.normal, viewDir, lightDir);
              var shadow_ray: Ray;
              shadow_ray.origin = sp.world_pos + sp.normal * 0.001;
              shadow_ray.direction = lightDir;
              var shadow_hit: Hit;
              let visible = !rayTrace(shadow_ray, &shadow_hit, true, light_dist - 0.5);
              if (visible) {
                p_hat = dot(brdf * cos_surface * ls.emission, vec3f(0.333));
              }
            } else if (ris_mode == 1u) {
              // BRDF: unshadowed integrand (p hat = BRDF * cos * Le)
              let brdf = evaluateBRDF(sp.material, sp.normal, viewDir, lightDir);
              p_hat = dot(brdf * cos_surface * ls.emission, vec3f(0.333));
            } else {
              // Geometric: cheap proxy (p hat = emission * cos_surface * cos_light / dist²)
              p_hat = ls.emission * cos_surface * cos_light / max(0.0001, dist_m * dist_m);
            }

            if (p_hat > 0.0) {
              let w = p_hat / q;
              res_w_sum += w;

              nee_seed = pcg_hash(nee_seed);
              if (f32(nee_seed) / f32(0xFFFFFFFFu) < w / res_w_sum) {
                res_u_draw = cand_u_draw;
                res_target_p = p_hat;
              }
            }
          }
        }

        // Store reservoir for temporal reuse (before shading, so next frame can pick up)
        let total_M = f32(M) + prev_res.w;
        if (use_restir && bounce == 0u) {
          reservoir[pix_idx] = vec4f(res_u_draw, res_w_sum, res_target_p, total_M);
        }

        // Can't reuse neeEmissiveTriangle here because it divides f by q(actual_point),
        // but RIS needs f / p_hat * w_sum / M. Going through neeEmissiveTriangle would
        // require multiplying back by q(centroid), introducing a q(centroid)/q(actual) bias.
        if (res_w_sum > 0.0 && res_target_p > 0.0) {
          let s1 = pcg_hash(nee_seed);
          let s2 = pcg_hash(s1);
          let u1 = f32(s1) / f32(0xFFFFFFFFu);
          let u2 = f32(s2) / f32(0xFFFFFFFFu);
          let ls = sampleEmissiveTriangle(res_u_draw, u1, u2);
          let lightVec = ls.position - sp.world_pos;
          let light_dist = length(lightVec);
          let lightDir = lightVec / light_dist;
          let cos_surface = max(0.0, dot(sp.normal, lightDir));
          let cos_light = max(0.0, dot(ls.normal, -lightDir));
          if (cos_surface > 0.0 && cos_light > 0.0) {
            var shadow_ray: Ray;
            shadow_ray.origin = sp.world_pos + sp.normal * 0.001;
            shadow_ray.direction = lightDir;
            var shadow_hit: Hit;
            if (!rayTrace(shadow_ray, &shadow_hit, true, light_dist - 0.5)) {
              let brdf = evaluateBRDF(sp.material, sp.normal, viewDir, lightDir);
              let f = brdf * cos_surface * ls.emission;
              let ris_M = select(f32(M), total_M, use_restir);
              let W = res_w_sum / (ris_M * res_target_p); // this line is why we need to inline
              color += throughput * f * W;
            }
          }
        }
      }
    }

    let seed_u = pcg_hash(px.x ^ pcg_hash(px.y ^ pcg_hash(bounce ^ u32(u.frame_count))));

    if (use_brdfis) {
      // BRDF hemisphere sampling (BRDF IS)
      let pick_seed = pcg_hash(seed_u ^ 0x13371337u);
      let pick = f32(pick_seed) / f32(0xFFFFFFFFu);
      let p_diffuse = 0.5 * (1.0 - sp.material.metalness); // since we have a diffuse lambertian and specular GGX, we randomly sample between the two depending on metalness

      var wi: vec3f;
      if (pick < p_diffuse) {
        let seed = bitcast<f32>(seed_u);
        wi = sampleSemiSphere(sp.normal, seed);
      } else {
        let halfVec_s = sampleGGXHalfVector(sp.normal, sp.material.roughness, seed_u);
        wi = reflect(-viewDir, halfVec_s);
        if (dot(wi, sp.normal) <= 0.0) { break; }
      }

      ray = Ray(sp.world_pos + sp.normal * 0.001, wi);

      let NdotL = max(0.0, dot(sp.normal, wi));
      let pdf   = brdfPdf(sp.material, sp.normal, viewDir, wi);
      if (pdf < 0.0001) { break; }

      throughput *= evaluateBRDF(sp.material, sp.normal, viewDir, wi) * NdotL / pdf;
      prev_pdf = pdf;
    } else {
      // Cosine-weighted hemisphere sampling (naive / NEE)
      let seed = bitcast<f32>(seed_u);
      let wi = sampleSemiSphere(sp.normal, seed);
      ray = Ray(sp.world_pos + sp.normal * 0.001, wi);

      let NdotL = max(0.0, dot(sp.normal, wi));
      let pdf   = cosinePdf(sp.normal, wi);
      if (pdf < 0.0001) { break; }
      throughput *= evaluateBRDF(sp.material, sp.normal, viewDir, wi) * NdotL / pdf;
      prev_pdf = pdf;
    }

    // Russian roulette
    let rr_seed = pcg_hash(seed_u ^ 0xDEADBEEFu);
    let rr_val = f32(rr_seed) / f32(0xFFFFFFFFu);
    let q = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 1.0);
    if (rr_val > q) { break; }
    throughput /= q;
  }

  // Touch unused bindings so auto layout keeps them alive
  let _uv = uvs[0];
  let _et = emissive_tris[0];
  let _rv = reservoir[0];

  color = applyBvhOverlay(screen_pos, color);

  let idx = px.y * u32(u.screen_width) + px.x;
  accum[idx] = accum[idx] + vec4f(color, 1.0);
}
`,Qe=`// === display.wgsl ===
// Reads from the accumulation buffer and displays the averaged result.

struct PointLight {
  position:  vec3f, intensity: f32,
  color:     vec3f, _pad:      f32,
  direction: vec3f, angle_rad: f32,
};

struct Material {
  baseColor: vec3f, roughness: f32,
  fresnel:   vec3f, metalness: f32,
  emission:  f32,   _pad1: f32, _pad2: f32, _pad3: f32,
};

struct Uniforms {
  camera_pos:      vec3f, fov_factor:    f32,
  camera_forward:  vec3f, aspect_ratio:  f32,
  camera_right:    vec3f, nb_lights:     f32,
  camera_up:       vec3f, frame_count:   f32,
  lights:          array<PointLight, 4>,
  nb_materials:    f32,
  bvh_vis_depth:   f32,
  bvh_heat_max:    f32,
  bvh_early_stop:  f32,
  screen_width:    f32,
  screen_height:   f32,
  render_mode:     f32,
  _pad2:           f32,
  materials:       array<Material, 16>,
};

@group(0) @binding(0) var<storage, read> accum: array<vec4f>;
@group(0) @binding(1) var<uniform> u: Uniforms;

struct VertexOutput {
  @builtin(position) clip_pos:   vec4f,
  @location(0)       screen_pos: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VertexOutput {
  let x = f32(i32(vi) / 2) * 4.0 - 1.0;
  let y = f32(i32(vi) % 2) * 4.0 - 1.0;
  var out: VertexOutput;
  out.clip_pos   = vec4f(x, y, 0.0, 1.0);
  out.screen_pos = vec2f(x, y);
  return out;
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f {
  let px = vec2u(input.clip_pos.xy);
  let pixel_value = accum[px.y * u32(u.screen_width) + px.x];
  return vec4f(pixel_value.rgb / pixel_value.w, 1.0);
}
`;function Je(t){return{position:[278,273,-800],target:[278,273,-799],up:[0,1,0],fov:Math.PI/3,aspect:t.width/t.height}}function he(t){const e=t.target[0]-t.position[0],n=t.target[1]-t.position[1],i=t.target[2]-t.position[2],r=Math.hypot(e,n,i),s=new Float32Array([e/r,n/r,i/r]),d=s[1]*t.up[2]-s[2]*t.up[1],c=s[2]*t.up[0]-s[0]*t.up[2],l=s[0]*t.up[1]-s[1]*t.up[0],a=Math.hypot(d,c,l),f=new Float32Array([d/a,c/a,l/a]),u=f[1]*s[2]-f[2]*s[1],g=f[2]*s[0]-f[0]*s[2],o=f[0]*s[1]-f[1]*s[0],p=Math.hypot(u,g,o),x=new Float32Array([u/p,g/p,o/p]);return{forward:s,right:f,up:x}}function ue(t,e,n){const i=t.target[0]-t.position[0],r=t.target[1]-t.position[1],s=t.target[2]-t.position[2],d=Math.hypot(i,r,s),c=i/d,l=r/d,a=s/d,f=l*t.up[2]-a*t.up[1],u=a*t.up[0]-c*t.up[2],g=c*t.up[1]-l*t.up[0],o=Math.hypot(f,u,g),p=f/o,x=u/o,m=g/o;t.position[0]+=p*e+t.up[0]*n,t.position[1]+=x*e+t.up[1]*n,t.position[2]+=m*e+t.up[2]*n,t.target[0]+=p*e+t.up[0]*n,t.target[1]+=x*e+t.up[1]*n,t.target[2]+=m*e+t.up[2]*n}function Ie(t,e){const{forward:n}=he(t);for(let i=0;i<3;i++)t.position[i]+=n[i]*e,t.target[i]+=n[i]*e}function ke(t,e){const{forward:n}=he(t),i=Math.hypot(t.target[0]-t.position[0],t.target[1]-t.position[1],t.target[2]-t.position[2]),r=Math.cos(e),s=Math.sin(e),d=n[0]*r+n[2]*s,c=-n[0]*s+n[2]*r;t.target[0]=t.position[0]+d*i,t.target[1]=t.position[1]+n[1]*i,t.target[2]=t.position[2]+c*i}function Le(t,e){const{forward:n,up:i}=he(t),r=Math.hypot(t.target[0]-t.position[0],t.target[1]-t.position[1],t.target[2]-t.position[2]),s=Math.cos(e),d=Math.sin(e),c=n[0]*s+i[0]*d,l=n[1]*s+i[1]*d,a=n[2]*s+i[2]*d,f=Math.hypot(c,l,a);t.target[0]=t.position[0]+c/f*r,t.target[1]=t.position[1]+l/f*r,t.target[2]=t.position[2]+a/f*r}function tn(t,e,n,i){t<i.minCorner[0]&&(i.minCorner[0]=t),e<i.minCorner[1]&&(i.minCorner[1]=e),n<i.minCorner[2]&&(i.minCorner[2]=n),t>i.maxCorner[0]&&(i.maxCorner[0]=t),e>i.maxCorner[1]&&(i.maxCorner[1]=e),n>i.maxCorner[2]&&(i.maxCorner[2]=n)}function en(t,e){const n=new Float32Array([1/0,1/0,1/0]),i=new Float32Array([-1/0,-1/0,-1/0]);for(let r=0;r<e.length;r++){const s=e[r];for(let d=0;d<3;d++){const c=t.indices[s*3+d],l=t.positions[c*3],a=t.positions[c*3+1],f=t.positions[c*3+2];tn(l,a,f,{minCorner:n,maxCorner:i})}}return{minCorner:n,maxCorner:i}}function De(t,e){const n=t.maxCorner[0]-t.minCorner[0],i=t.maxCorner[1]-t.minCorner[1],r=t.maxCorner[2]-t.minCorner[2];return(n*(i+r)+i*r)*e}function Te(t,e,n){const i=e.length;let r=1e30,s=-1e30;const d=new Float32Array(i);for(let M=0;M<i;M++){const h=e[M],S=t.indices[h*3],B=t.indices[h*3+1],Y=t.indices[h*3+2],nt=(t.positions[S*3+n]+t.positions[B*3+n]+t.positions[Y*3+n])/3;d[M]=nt,nt<r&&(r=nt),nt>s&&(s=nt)}const c=12;if(s===r){const M=Math.floor(i/2);return{left:e.slice(0,M),right:e.slice(M),cost:1e30}}const l=new Uint32Array(c),a=new Float32Array(c*3).fill(1/0),f=new Float32Array(c*3).fill(-1/0),u=c/(s-r),g=new Uint8Array(i);for(let M=0;M<i;M++){const h=Math.min(Math.max(Math.floor((d[M]-r)*u),0),c-1);g[M]=h,l[h]++;const S=e[M],B=h*3;for(let Y=0;Y<3;Y++){const nt=t.indices[S*3+Y],z=t.positions[nt*3],T=t.positions[nt*3+1],O=t.positions[nt*3+2];z<a[B]&&(a[B]=z),T<a[B+1]&&(a[B+1]=T),O<a[B+2]&&(a[B+2]=O),z>f[B]&&(f[B]=z),T>f[B+1]&&(f[B+1]=T),O>f[B+2]&&(f[B+2]=O)}}const o=new Float32Array(c*3).fill(1/0),p=new Float32Array(c*3).fill(-1/0),x=new Uint32Array(c),m=(c-1)*3;o[m]=a[m],o[m+1]=a[m+1],o[m+2]=a[m+2],p[m]=f[m],p[m+1]=f[m+1],p[m+2]=f[m+2],x[c-1]=l[c-1];for(let M=c-2;M>=0;M--){const h=M*3,S=h+3;o[h]=o[S],o[h+1]=o[S+1],o[h+2]=o[S+2],p[h]=p[S],p[h+1]=p[S+1],p[h+2]=p[S+2],l[M]>0&&(a[h]<o[h]&&(o[h]=a[h]),a[h+1]<o[h+1]&&(o[h+1]=a[h+1]),a[h+2]<o[h+2]&&(o[h+2]=a[h+2]),f[h]>p[h]&&(p[h]=f[h]),f[h+1]>p[h+1]&&(p[h+1]=f[h+1]),f[h+2]>p[h+2]&&(p[h+2]=f[h+2])),x[M]=x[M+1]+l[M]}let v=1,H=1e30;{let M=a[0],h=a[1],S=a[2],B=f[0],Y=f[1],nt=f[2],z=l[0];for(let T=1;T<c;T++){const O=T*3,it=B-M,rt=Y-h,$=nt-S,X=p[O]-o[O],P=p[O+1]-o[O+1],st=p[O+2]-o[O+2],Q=(it*(rt+$)+rt*$)*z+(X*(P+st)+P*st)*x[T];Q<H&&(H=Q,v=T),l[T]>0&&(a[O]<M&&(M=a[O]),a[O+1]<h&&(h=a[O+1]),a[O+2]<S&&(S=a[O+2]),f[O]>B&&(B=f[O]),f[O+1]>Y&&(Y=f[O+1]),f[O+2]>nt&&(nt=f[O+2])),z+=l[T]}}let U=0;for(let M=0;M<i;M++)g[M]<v&&U++;const A=new Uint32Array(U),D=new Uint32Array(i-U);let R=0,K=0;for(let M=0;M<i;M++)g[M]<v?A[R++]=e[M]:D[K++]=e[M];return{left:A,right:D,cost:H}}function nn(t,e){let n=Te(t,e,0);for(const i of[1,2]){const r=Te(t,e,i);r.cost<n.cost&&(n=r)}return{partition:[n.left,n.right],cost:n.cost}}function sn(t){const n=[],i=[];let r=0;function s(l,a){const u=r++*8,g=en(t,l);function o(){const H=i.length;for(const U of l)i.push(U);n[u]=g.minCorner[0],n[u+1]=g.minCorner[1],n[u+2]=g.minCorner[2],n[u+3]=l.length,n[u+4]=g.maxCorner[0],n[u+5]=g.maxCorner[1],n[u+6]=g.maxCorner[2],n[u+7]=H}if(l.length<=1||a>=32){o();return}const{partition:p,cost:x}=nn(t,l),[m,v]=p;if(m.length===0||v.length===0){o();return}if(x+De(g,1)>De(g,l.length)){o();return}n[u]=g.minCorner[0],n[u+1]=g.minCorner[1],n[u+2]=g.minCorner[2],n[u+3]=0,n[u+4]=g.maxCorner[0],n[u+5]=g.maxCorner[1],n[u+6]=g.maxCorner[2],s(m,a+1),n[u+7]=r,s(v,a+1)}const d=t.indices.length/3,c=new Uint32Array(d);for(let l=0;l<d;l++)c[l]=l;return s(c,0),{data:n,nodeCount:r,primitives:new Uint32Array(i)}}function rn(t){const n=[];let i=0;function r(s){const c=i++*8,l=[1e30,1e30,1e30],a=[-1e30,-1e30,-1e30];for(const o of s)for(let p=0;p<3;p++)t[o].minCorner[p]<l[p]&&(l[p]=t[o].minCorner[p]),t[o].maxCorner[p]>a[p]&&(a[p]=t[o].maxCorner[p]);if(n[c]=l[0],n[c+1]=l[1],n[c+2]=l[2],n[c+4]=a[0],n[c+5]=a[1],n[c+6]=a[2],s.length===1){n[c+3]=1,n[c+7]=s[0];return}const f=[a[0]-l[0],a[1]-l[1],a[2]-l[2]].indexOf(Math.max(a[0]-l[0],a[1]-l[1],a[2]-l[2])),u=s.slice().sort((o,p)=>{const x=(t[o].minCorner[f]+t[o].maxCorner[f])*.5,m=(t[p].minCorner[f]+t[p].maxCorner[f])*.5;return x-m}),g=Math.floor(u.length/2);n[c+3]=0,r(u.slice(0,g)),n[c+7]=i,r(u.slice(g))}return r(Array.from({length:t.length},(s,d)=>d)),{data:n,nodeCount:i}}function qe(t,e){return[t[1]*e[2]-t[2]*e[1],t[2]*e[0]-t[0]*e[2],t[0]*e[1]-t[1]*e[0]]}function de(t,e){return[t[0]-e[0],t[1]-e[1],t[2]-e[2]]}function on(t){const e=Math.sqrt(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]);return[t[0]/e,t[1]/e,t[2]/e]}function te(){return new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])}function fe(t,e){const n=new Float32Array(16);for(let i=0;i<4;i++)for(let r=0;r<4;r++)n[i*4+r]=t[0+r]*e[i*4+0]+t[4+r]*e[i*4+1]+t[8+r]*e[i*4+2]+t[12+r]*e[i*4+3];return n}function Ge(t,e,n){const i=te();return i[12]=t,i[13]=e,i[14]=n,i}function an(t){const e=te();return e[0]=t,e[5]=t,e[10]=t,e}function ln(t){const e=Math.cos(t),n=Math.sin(t),i=te();return i[5]=e,i[9]=-n,i[6]=n,i[10]=e,i}function cn(t){const e=Math.cos(t),n=Math.sin(t),i=te();return i[0]=e,i[8]=n,i[2]=-n,i[10]=e,i}function un(t){const e=Math.cos(t),n=Math.sin(t),i=te();return i[0]=e,i[4]=-n,i[1]=n,i[5]=e,i}function fn(t){const e=t[0]*t[0]+t[1]*t[1]+t[2]*t[2],n=new Float32Array(16);return n[0]=t[0]/e,n[4]=t[1]/e,n[8]=t[2]/e,n[1]=t[4]/e,n[5]=t[5]/e,n[9]=t[6]/e,n[2]=t[8]/e,n[6]=t[9]/e,n[10]=t[10]/e,n[12]=-(n[0]*t[12]+n[4]*t[13]+n[8]*t[14]),n[13]=-(n[1]*t[12]+n[5]*t[13]+n[9]*t[14]),n[14]=-(n[2]*t[12]+n[6]*t[13]+n[10]*t[14]),n[15]=1,n}function dn({translation:[t,e,n],rotation:[i,r,s],scale:d}){return fe(Ge(t,e,n),fe(cn(r),fe(ln(i),fe(un(s),an(d)))))}function hn(t,e,n,i){const r=[],s=[],d=[],c=[],l=[];for(let a=0;a<=e;a++){const f=a*Math.PI/e,u=Math.sin(f),g=Math.cos(f);for(let o=0;o<=n;o++){const p=o*2*Math.PI/n,x=Math.cos(p)*u,m=g,v=Math.sin(p)*u;r.push(t*x,t*m,t*v),s.push(x,m,v),d.push(o/n,1-a/e),c.push(i[0],i[1],i[2])}}for(let a=0;a<e;a++)for(let f=0;f<n;f++){const u=a*(n+1)+f,g=u+n+1;l.push(u,u+1,g,g,u+1,g+1)}return{positions:new Float32Array(r),normals:new Float32Array(s),uvs:new Float32Array(d),colors:new Float32Array(c),indices:new Uint32Array(l)}}function Ot(t,e,n,i,r){const s=on(qe(de(t,e),de(t,i)));return{positions:new Float32Array([...t,...e,...n,...i]),normals:new Float32Array([...s,...s,...s,...s]),uvs:new Float32Array([0,0,1,0,1,1,0,1]),colors:new Float32Array([...r,...r,...r,...r]),indices:new Uint32Array([0,1,2,0,2,3])}}function pn(t){const e=[],n=[],i=[],r=[],s=[];let d=0;for(const c of t){for(let l=0;l<c.positions.length;l++)e.push(c.positions[l]);for(let l=0;l<c.normals.length;l++)n.push(c.normals[l]);for(let l=0;l<c.uvs.length;l++)i.push(c.uvs[l]);for(let l=0;l<c.colors.length;l++)r.push(c.colors[l]);for(let l=0;l<c.indices.length;l++)s.push(c.indices[l]+d);d+=c.positions.length/3}return{positions:new Float32Array(e),normals:new Float32Array(n),uvs:new Float32Array(i),colors:new Float32Array(r),indices:new Uint32Array(s)}}function mn(t,e){const n=new Float32Array(t.positions.length),i=new Float32Array(t.normals.length);for(let r=0;r<t.positions.length;r+=3){const s=t.positions[r],d=t.positions[r+1],c=t.positions[r+2];n[r]=e[0]*s+e[4]*d+e[8]*c+e[12],n[r+1]=e[1]*s+e[5]*d+e[9]*c+e[13],n[r+2]=e[2]*s+e[6]*d+e[10]*c+e[14]}for(let r=0;r<t.normals.length;r+=3){const s=t.normals[r],d=t.normals[r+1],c=t.normals[r+2];let l=e[0]*s+e[4]*d+e[8]*c,a=e[1]*s+e[5]*d+e[9]*c,f=e[2]*s+e[6]*d+e[10]*c;const u=Math.sqrt(l*l+a*a+f*f);u>1e-4&&(l/=u,a/=u,f/=u),i[r]=l,i[r+1]=a,i[r+2]=f}return{positions:n,normals:i,uvs:t.uvs,colors:t.colors,indices:t.indices}}function Ue(t,e=[1,1,1]){const n=[],i=[],r=[],s=[],d=[],c=[],l=[];for(const u of t.split(`
`)){const g=u.trim().split(/\s+/);if(g[0]==="v")n.push(+g[1],+g[2],+g[3]);else if(g[0]==="vn")i.push(+g[1],+g[2],+g[3]);else if(g[0]==="vt")r.push(+g[1],+g[2]);else if(g[0]==="f"){const o=g.slice(1);for(let p=1;p<o.length-1;p++)for(const x of[o[0],o[p],o[p+1]]){const[m,v,H]=x.split("/"),U=(parseInt(m)-1)*3;if(s.push(n[U],n[U+1],n[U+2]),H){const A=(parseInt(H)-1)*3;d.push(i[A],i[A+1],i[A+2])}else d.push(0,0,0);if(v){const A=(parseInt(v)-1)*2;c.push(r[A],r[A+1])}else c.push(0,0);l.push(e[0],e[1],e[2])}}}if(i.length===0)for(let u=0;u<s.length;u+=9){const g=s[u+3]-s[u],o=s[u+4]-s[u+1],p=s[u+5]-s[u+2],x=s[u+6]-s[u],m=s[u+7]-s[u+1],v=s[u+8]-s[u+2],H=o*v-p*m,U=p*x-g*v,A=g*m-o*x,D=Math.sqrt(H*H+U*U+A*A)||1;for(let R=0;R<3;R++)d[u+R*3]=H/D,d[u+R*3+1]=U/D,d[u+R*3+2]=A/D}const a=s.length/3,f=new Uint32Array(a);for(let u=0;u<a;u++)f[u]=u;return{positions:new Float32Array(s),normals:new Float32Array(d),uvs:new Float32Array(c),colors:new Float32Array(l),indices:f}}const Pe={cornell:{position:[278,273,-800],target:[278,273,-799]},sponza:{position:[0,400,0],target:[1,400,0]}};async function gn(t,e={}){return t==="sponza"?vn(e):yn(e)}function _n(t){return function(){t|=0,t=t+1831565813|0;let e=Math.imul(t^t>>>15,1|t);return e=e+Math.imul(e^e>>>7,61|e)^e,((e^e>>>14)>>>0)/4294967296}}async function vn(t={}){const e=t.lightCount??100,n=await fetch("/assets/sponza.obj").then(f=>f.text()),i=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),r={stone:{id:0,diffuseAlbedo:[.76,.7,.58],roughness:.7,metalness:0,fresnel:[.04,.04,.04],emission:0},light:{id:1,diffuseAlbedo:[1,.85,.6],roughness:0,metalness:0,fresnel:[.04,.04,.04],emission:30}},s=hn(15,8,8,[1,.85,.6]),d=_n(42),c=[];for(let f=0;f<e;f++){const u=(d()*2-1)*1500,g=50+d()*1150,o=(d()*2-1)*900;c.push(mn(s,Ge(u,g,o)))}const l=pn(c);return{objects:[{mesh:Ue(n,[.76,.7,.58]),material:r.stone,transform:i,label:"Sponza"},{mesh:l,material:r.light,transform:i,label:"Lights"}],lights:[]}}async function yn(t={}){const e=t.dragonRes||"2348",n=await fetch(`/assets/dragon_${e}.obj`).then(f=>f.text()),i=new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),r={white:{id:0,diffuseAlbedo:[1,1,1],roughness:0,metalness:0,fresnel:[.05,.05,.05],emission:0},red:{id:1,diffuseAlbedo:[.65,.05,.05],roughness:0,metalness:0,fresnel:[.05,.05,.05],emission:0},green:{id:2,diffuseAlbedo:[.12,.45,.15],roughness:0,metalness:0,fresnel:[.05,.05,.05],emission:0},light:{id:3,diffuseAlbedo:[1,1,1],roughness:0,metalness:0,fresnel:[.05,.05,.05],emission:30},dragon:{id:4,diffuseAlbedo:[1,.71,.29],roughness:.3,metalness:1,fresnel:[1,.71,.29],emission:0}},s=[1,1,1],d=[.65,.05,.05],c=[.12,.45,.15];return{objects:[{mesh:Ot([552.8,0,0],[0,0,0],[0,0,559.2],[549.6,0,559.2],s),material:r.white,transform:i,label:"Floor"},{mesh:Ot([556,548.8,0],[556,548.8,559.2],[0,548.8,559.2],[0,548.8,0],s),material:r.white,transform:i,label:"Ceiling"},{mesh:Ot([549.6,0,559.2],[0,0,559.2],[0,548.8,559.2],[556,548.8,559.2],s),material:r.white,transform:i,label:"Back wall"},{mesh:Ot([0,0,559.2],[0,0,0],[0,548.8,0],[0,548.8,559.2],c),material:r.green,transform:i,label:"Right wall"},{mesh:Ot([552.8,0,0],[549.6,0,559.2],[556,548.8,559.2],[556,548.8,0],d),material:r.red,transform:i,label:"Left wall"},{mesh:Ot([556,548.8,0],[0,548.8,0],[0,0,0],[552.8,0,0],s),material:r.white,transform:i,label:"Front wall"},{mesh:Ot([343,548,227],[343,548,332],[213,548,332],[213,548,227],s),material:r.light,transform:i,label:"Light"},{mesh:Ue(n),material:r.dragon,transform:dn({translation:[279,115,269],rotation:[0,Math.PI/4,0],scale:2}),label:"Dragon"}],lights:[]}}function xn(t,e,n,i){const r=[];for(let a=0;a<t.instances.length;a++){const f=t.instances[a],u=t.materials[f.materialId];if(!u.emission||u.emission<=0)continue;const o=t.meshes[f.meshIndex].indices.length/3,p=i[f.meshIndex];for(let x=0;x<o;x++){const m=p+x,v=e[m*3],H=e[m*3+1],U=e[m*3+2],A=[n[v*3],n[v*3+1],n[v*3+2]],D=[n[H*3],n[H*3+1],n[H*3+2]],R=[n[U*3],n[U*3+1],n[U*3+2]],K=de(D,A),M=de(R,A),h=qe(K,M),B=.5*Math.sqrt(h[0]*h[0]+h[1]*h[1]+h[2]*h[2])/1e6,Y=u.emission*B;r.push({triIndex:m,instanceIndex:a,area:B,power:Y,cdf:0})}}let s=0;for(const a of r)s+=a.power,a.cdf=s;const d=new ArrayBuffer(r.length*16),c=new Uint32Array(d),l=new Float32Array(d);for(let a=0;a<r.length;a++)c[a*4]=r[a].triIndex,c[a*4+1]=r[a].instanceIndex,l[a*4+2]=r[a].area,l[a*4+3]=r[a].cdf;return console.log(`Emissive triangles: ${r.length}, total power: ${s.toFixed(2)}`),{buffer:d,count:r.length,totalPower:s}}function bn(t){const e=new Map;for(const c of t.objects)e.set(c.material.id,c.material);const n=e.size===0?-1:Math.max(...e.keys()),i=[];for(let c=0;c<=n;c++)i.push(e.get(c)??{id:c,diffuseAlbedo:[0,0,0],roughness:0,metalness:0,fresnel:[0,0,0],emission:0});const r=new Map,s=[],d=[];for(const c of t.objects){let l=r.get(c.mesh);l===void 0&&(l=s.length,r.set(c.mesh,l),s.push({positions:c.mesh.positions,normals:c.mesh.normals,uvs:c.mesh.uvs,colors:c.mesh.colors,indices:c.mesh.indices})),d.push({meshIndex:l,transform:c.transform,materialId:c.material.id,label:c.label??""})}return{meshes:s,instances:d,materials:i}}async function wn(t){if(!navigator.gpu)throw new Error("WebGPU not supported");const e=await navigator.gpu.requestAdapter();if(!e)throw new Error("No GPU adapter found");const n=e.features.has("timestamp-query"),i=await e.requestDevice({requiredFeatures:n?["timestamp-query"]:[],requiredLimits:{maxStorageBuffersPerShaderStage:Math.min(e.limits.maxStorageBuffersPerShaderStage,16)}}),r=t.getContext("webgpu"),s=navigator.gpu.getPreferredCanvasFormat();return r.configure({device:i,format:s,alphaMode:"opaque"}),{device:i,context:r,format:s,hasTimestamps:n}}function An(t,e){if(!e)return null;const n=t.createQuerySet({type:"timestamp",count:2}),i=t.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),r=[0,1].map(()=>t.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}));return{querySet:n,resolveBuffer:i,readBuffers:r,readIndex:0}}function ze(t,e,n){return t.createBuffer({size:e*n*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST})}const je=268;function Cn(t){return t.createBuffer({size:je*4,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST})}function Mn(t,e,n,i,r=-1,s=4,d=0,c=1024,l=1024,a=0,f=0,u=1,g=1){const o=new Float32Array(je),p=he(t),x=Math.tan(t.fov/2);o[0]=t.position[0],o[1]=t.position[1],o[2]=t.position[2],o[3]=x,o[4]=p.forward[0],o[5]=p.forward[1],o[6]=p.forward[2],o[7]=t.aspect,o[8]=p.right[0],o[9]=p.right[1],o[10]=p.right[2],o[11]=e.length,o[12]=p.up[0],o[13]=p.up[1],o[14]=p.up[2],o[15]=i||0;for(let m=0;m<e.length;m++){const v=16+m*12;o[v]=e[m].position[0],o[v+1]=e[m].position[1],o[v+2]=e[m].position[2],o[v+3]=e[m].intensity,o[v+4]=e[m].color[0],o[v+5]=e[m].color[1],o[v+6]=e[m].color[2],o[v+7]=0,o[v+8]=e[m].direction[0],o[v+9]=e[m].direction[1],o[v+10]=e[m].direction[2],o[v+11]=e[m].angle*Math.PI/180}o[64]=n.length,o[65]=r,o[66]=s,o[67]=d,o[68]=c,o[69]=l,o[70]=a,o[71]=f,o[72]=u,o[73]=g;for(let m=0;m<n.length;m++){const v=76+m*12;o[v]=n[m].diffuseAlbedo[0],o[v+1]=n[m].diffuseAlbedo[1],o[v+2]=n[m].diffuseAlbedo[2],o[v+3]=n[m].roughness||0,o[v+4]=n[m].fresnel[0],o[v+5]=n[m].fresnel[1],o[v+6]=n[m].fresnel[2],o[v+7]=n[m].metalness||0,o[v+8]=n[m].emission||0}return o}function Fn(t,e,n){const i=t.createShaderModule({code:n});return t.createRenderPipeline({layout:"auto",vertex:{module:i,entryPoint:"vs"},fragment:{module:i,entryPoint:"fs",targets:[{format:e}]}})}function Re(t,e){const n=t.createShaderModule({code:e});return t.createComputePipeline({layout:"auto",compute:{module:n,entryPoint:"computeMain"}})}function Sn(t,e,n,i,r,s){const d=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,c=t.createBuffer({label:"vertices",size:e.positions.byteLength,usage:d}),l=t.createBuffer({label:"indices",size:e.indices.byteLength,usage:d}),a=t.createBuffer({label:"normals",size:e.normals.byteLength,usage:d}),f=t.createBuffer({label:"uvs",size:e.uvs.byteLength,usage:d}),u=t.createBuffer({label:"tlas",size:n.byteLength,usage:d}),g=t.createBuffer({label:"blases",size:i.byteLength,usage:d}),o=t.createBuffer({label:"instances",size:r.byteLength,usage:d}),p=t.createBuffer({label:"emissive",size:Math.max(s.byteLength,16),usage:d});return t.queue.writeBuffer(c,0,e.positions),t.queue.writeBuffer(l,0,e.indices),t.queue.writeBuffer(a,0,e.normals),t.queue.writeBuffer(f,0,e.uvs),t.queue.writeBuffer(u,0,n),t.queue.writeBuffer(g,0,i),t.queue.writeBuffer(o,0,r),s.byteLength>0&&t.queue.writeBuffer(p,0,s),{vertexBuffer:c,indexBuffer:l,normalBuffer:a,uvBuffer:f,tlasBuffer:u,blasBuffer:g,instanceBuffer:o,emissiveBuffer:p}}function Bn(t){if(t)for(const e of Object.keys(t))t[e].destroy()}function Oe(t,e,n,i,r,s){return t.createBindGroup({layout:e.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:i.vertexBuffer}},{binding:2,resource:{buffer:i.indexBuffer}},{binding:3,resource:{buffer:i.normalBuffer}},{binding:4,resource:{buffer:i.uvBuffer}},{binding:5,resource:{buffer:i.tlasBuffer}},{binding:6,resource:{buffer:i.blasBuffer}},{binding:7,resource:{buffer:i.instanceBuffer}},{binding:8,resource:{buffer:r}},{binding:9,resource:{buffer:i.emissiveBuffer}},{binding:10,resource:{buffer:s}}]})}function En(t,e,n,i){return t.createBindGroup({layout:e.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:n}},{binding:1,resource:{buffer:i}}]})}function In(t,e,n,i,r,s,d,c,l){const a=t.createCommandEncoder(),f={};l&&(f.timestampWrites={querySet:l.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1});const u=a.beginComputePass(f);u.setPipeline(i),u.setBindGroup(0,s),u.dispatchWorkgroups(Math.ceil(d/8),Math.ceil(c/8)),u.end();const g={colorAttachments:[{view:e.getCurrentTexture().createView(),loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]},o=a.beginRenderPass(g);if(o.setPipeline(n),o.setBindGroup(0,r),o.draw(3),o.end(),l){a.resolveQuerySet(l.querySet,0,2,l.resolveBuffer,0);const p=l.readBuffers[l.readIndex];p.mapState==="unmapped"&&a.copyBufferToBuffer(l.resolveBuffer,0,p,0,16)}if(t.queue.submit([a.finish()]),l){const p=l.readBuffers[l.readIndex];l.readIndex=(l.readIndex+1)%2,p.mapState==="unmapped"&&p.mapAsync(GPUMapMode.READ).then(()=>{const x=new BigInt64Array(p.getMappedRange()),m=Number(x[1]-x[0])/1e6;p.unmap(),l.onGpuTime&&l.onGpuTime(m)})}}function $t(t,e={}){const n=e.width||520,i=e.height||300,r=e.roughness??.4,s=e.metalness??1,d=e.baseColor??[.8,.8,.8],c=e.fresnel??[.04,.04,.04],l=e.showLobe??!0,a=e.showSamples??!1;let f=e.numSamples||9;const u=e.sampling||"uniform",g=e.showSlider??!1,o=e.showLight??!0,p=e.showLightRay??!0,x=e.swatchLightAngle!=null?e.swatchLightAngle*Math.PI/180:null,m=e.areaLight??!1,v=e.areaLightWidth??80,H=e.showSwatch??!0,U=e.showNeeRays??!1,A=e.crossOutAreaHits??!1,D=e.draggableLight??!0,R=e.misWeights??!1,K=e.misWeightMode??"length",M=e.neeRayCount??3,h=n/2,S=i*.78,B=S,Y=e.arrowLen??120,nt=110,z=150,T=130,O=12;let it=-(e.incomingAngle??55)*Math.PI/180,rt=(e.lightAngle??30)*Math.PI/180;const $=document.createElementNS.bind(document,"http://www.w3.org/2000/svg"),X=Math.ceil(S+20),P=$("svg");P.setAttribute("width",String(n)),P.setAttribute("height",String(X)),P.setAttribute("viewBox",`0 0 ${n} ${X}`),P.style.display="block",P.style.margin="16px auto",P.style.cursor="default",P.style.userSelect="none";const st=$("defs");P.appendChild(st);const Q="viz"+Math.random().toString(36).slice(2,6);for(const[y,_,F]of[["cam","#333",1],["light","#e67e22",.8]]){const I=$("marker");I.setAttribute("id",`${Q}-ah-${y}`),I.setAttribute("markerWidth","8"),I.setAttribute("markerHeight","6"),I.setAttribute("refX","7"),I.setAttribute("refY","3"),I.setAttribute("orient","auto");const C=$("polygon");C.setAttribute("points","0 0, 8 3, 0 6"),C.setAttribute("fill",_),C.setAttribute("opacity",String(F)),I.appendChild(C),st.appendChild(I)}const at=$("marker");at.setAttribute("id",`${Q}-ah-nee`),at.setAttribute("markerWidth","8"),at.setAttribute("markerHeight","6"),at.setAttribute("refX","7"),at.setAttribute("refY","3"),at.setAttribute("orient","auto");const gt=$("polygon");gt.setAttribute("points","0 0, 8 3, 0 6"),gt.setAttribute("fill","#f1c40f"),at.appendChild(gt),st.appendChild(at);const mt=$("marker");mt.setAttribute("id",`${Q}-ah-sample`),mt.setAttribute("markerWidth","8"),mt.setAttribute("markerHeight","6"),mt.setAttribute("refX","7"),mt.setAttribute("refY","3"),mt.setAttribute("orient","auto");const bt=$("polygon");bt.setAttribute("points","0 0, 8 3, 0 6"),bt.setAttribute("fill","rgba(60,60,60,0.55)"),mt.appendChild(bt),st.appendChild(mt);const At=$("g"),Et=$("line");Object.entries({x1:20,y1:B,x2:n-20,y2:B,stroke:"#e67e22","stroke-width":3}).forEach(([y,_])=>Et.setAttribute(y,String(_))),At.appendChild(Et);for(let y=30;y<n-30;y+=14){const _=$("line");Object.entries({x1:y,y1:B+2,x2:y-10,y2:B+14,stroke:"#e67e22","stroke-width":1.5}).forEach(([F,I])=>_.setAttribute(F,String(I))),At.appendChild(_)}P.appendChild(At);const kt=$("line");Object.entries({x1:h,y1:B,x2:h,y2:B-140,stroke:"#999","stroke-width":1,"stroke-dasharray":"4 3"}).forEach(([y,_])=>kt.setAttribute(y,String(_))),P.appendChild(kt);const _t=$("text");if(_t.setAttribute("x",String(h+5)),_t.setAttribute("y",String(B-142)),_t.setAttribute("font-size","12"),_t.setAttribute("fill","#999"),_t.setAttribute("font-family","serif"),_t.setAttribute("font-style","italic"),_t.textContent="n",P.appendChild(_t),m){const y=v,_=12,F=h-y/2,I=B-185,C=$("text");C.setAttribute("x",String(F+y/2)),C.setAttribute("y",String(I-6)),C.setAttribute("text-anchor","middle"),C.setAttribute("font-size","11"),C.setAttribute("font-family","monospace"),C.setAttribute("fill","#c87800"),C.textContent="area light";const b=$("rect");Object.entries({x:F,y:I,width:y,height:_,rx:2,fill:"#ffe066",stroke:"#f39c12","stroke-width":2}).forEach(([E,G])=>b.setAttribute(E,String(G))),P.appendChild(C),P.appendChild(b);for(let E=0;E<=4;E++){const G=F+E/4*y,w=$("line");Object.entries({x1:G,y1:I+_,x2:G,y2:I+_+10,stroke:"#ffe066","stroke-width":1.5,opacity:"0.8"}).forEach(([L,N])=>w.setAttribute(L,String(N))),P.appendChild(w)}}const lt=$("g");P.appendChild(lt);const Lt=40,jt=n-60,Ht=30;let Z,vt;H&&(Z=$("rect"),Object.entries({x:jt,y:Ht,width:Lt,height:Lt,rx:4,stroke:"#ccc","stroke-width":1}).forEach(([y,_])=>Z.setAttribute(y,String(_))),P.appendChild(Z),vt=$("text"),vt.setAttribute("x",String(jt+Lt/2)),vt.setAttribute("y",String(Ht+Lt+14)),vt.setAttribute("text-anchor","middle"),vt.setAttribute("font-size","11"),vt.setAttribute("fill","#666"),vt.setAttribute("font-family","monospace"),P.appendChild(vt));function Vt(y){const _=r*r,F=_*_,I=y*y*(F-1)+1;return F/Math.max(1e-7,Math.PI*I*I)}function Dt(y){const F=r*r*Math.sqrt(2/Math.PI);return y/Math.max(1e-4,y*(1-F)+F)}function Wt(y){const F=Math.max(0,1-y)**5;return c.map((I,C)=>{const b=I*(1-s)+d[C]*s;return b+(1-b)*F})}function ee(y){const _=Wt(y);return(_[0]+_[1]+_[2])/3}function qt(y,_){const F=[Math.sin(_),Math.cos(_)],I=Math.max(0,F[1]),C=Math.max(0,y[1]);if(I<=0||C<=0)return 0;let b=[y[0]+F[0],y[1]+F[1]];const E=Math.sqrt(b[0]*b[0]+b[1]*b[1]);if(E<1e-8)return 0;b=[b[0]/E,b[1]/E];const G=Math.max(0,b[1]),w=Math.max(0,b[0]*y[0]+b[1]*y[1]);if(G<=0||w<=0)return 0;const L=Vt(G),N=Dt(C)*Dt(I),V=ee(w),W=L*V*N/Math.max(1e-4,4*C*I),ct=(d[0]+d[1]+d[2])/3,pt=(1-s)*(1-V)*ct/Math.PI;return W+pt}function pe(y,_){const F=Math.max(0,_[1]),I=Math.max(0,y[1]);if(F<=0||I<=0)return[0,0,0];let C=[y[0]+_[0],y[1]+_[1]];const b=Math.sqrt(C[0]*C[0]+C[1]*C[1]);if(b<1e-8)return[0,0,0];C=[C[0]/b,C[1]/b];const E=Math.max(0,C[1]),G=Math.max(0,C[0]*y[0]+C[1]*y[1]);if(E<=0||G<=0)return[0,0,0];const w=Vt(E),L=Dt(I)*Dt(F),N=Wt(G),V=Math.max(1e-4,4*I*F),W=ct=>Math.min(255,Math.round(ct*255));return[0,1,2].map(ct=>{const ht=w*N[ct]*L/V*F,pt=d[ct]/Math.PI*F,q=(1-s)*(1-N[ct]);return W((q*pt+ht)*1.75)})}function Xt(y,_){const F=Math.max(0,Math.cos(y));if(F<=0)return 0;const I=[Math.sin(y),Math.cos(y)];let C=[_[0]+I[0],_[1]+I[1]];const b=Math.sqrt(C[0]*C[0]+C[1]*C[1]);if(b<1e-8)return 0;C=[C[0]/b,C[1]/b];const E=Math.max(0,C[1]),G=Math.max(0,_[0]*C[0]+_[1]*C[1]),w=Vt(E),L=.5*(1-s);return L*F/Math.PI+(1-L)*w*E/Math.max(1e-4,4*G)}function Yt(y){if(!m||v<=0)return 0;const _=Math.cos(y);if(_<=0)return 0;const F=185/_,I=h+Math.sin(y)*F,C=h-v/2,b=h+v/2;if(I<C||I>b)return 0;const E=Math.atan2(C-h,185),w=Math.atan2(b-h,185)-E;return w<=0?0:1/w}function Zt(y,_,F,I){let C=_,b=F,E=I;return K==="length"?C=_*Math.max(.1,y):K==="opacity"?b=.1+.8*y:K==="thickness"&&(E=.5+2.5*y),{len:C,opacity:b,width:E}}function Ft(){for(;lt.firstChild;)lt.removeChild(lt.firstChild);const y=[-Math.sin(it),Math.cos(it)],_=[Math.sin(rt),Math.cos(rt)],F=Math.sin(it),I=Math.cos(it),C=$("line");if(Object.entries({x1:h-F*z,y1:S-I*z,x2:h,y2:S,stroke:"#333","stroke-width":2.5,"marker-end":`url(#${Q}-ah-cam)`}).forEach(([b,E])=>C.setAttribute(b,String(E))),lt.appendChild(C),o){const b=h+Math.sin(rt)*T,E=S-Math.cos(rt)*T;if(p){const w=$("line");Object.entries({x1:b,y1:E,x2:h,y2:S,stroke:"#e67e22","stroke-width":2,"stroke-dasharray":"6 3","marker-end":`url(#${Q}-ah-light)`}).forEach(([L,N])=>w.setAttribute(L,String(N))),lt.appendChild(w)}const G=$("circle");Object.entries({cx:b,cy:E,r:O,fill:"#f39c12",stroke:"#e67e22","stroke-width":2,cursor:D?"grab":"default"}).forEach(([w,L])=>G.setAttribute(w,String(L))),lt.appendChild(G);for(let w=0;w<8;w++){const L=w/8*Math.PI*2,N=$("line");Object.entries({x1:b+Math.cos(L)*(O+2),y1:E+Math.sin(L)*(O+2),x2:b+Math.cos(L)*(O+7),y2:E+Math.sin(L)*(O+7),stroke:"#f39c12","stroke-width":1.5,"stroke-linecap":"round"}).forEach(([V,W])=>N.setAttribute(V,String(W))),lt.appendChild(N)}}if(l){const E=[];for(let w=0;w<=200;w++){const L=-Math.PI/2+Math.PI*w/200,N=Math.log2(1+20*qt(y,L))/Math.log2(21),V=Math.sin(L),W=-Math.cos(L);E.push(`${h+V*N*nt},${S+W*N*nt}`)}const G=$("polyline");G.setAttribute("points",E.join(" ")),Object.entries({fill:"none",stroke:"#27ae60","stroke-width":2.5,"stroke-dasharray":"8 4"}).forEach(([w,L])=>G.setAttribute(w,String(L))),lt.appendChild(G)}if(a){let b=[];if(u==="uniform")for(let w=0;w<f;w++){const L=(w+.5)/f;b.push(-Math.PI/2+L*Math.PI)}else if(u==="cosine")for(let w=0;w<f;w++){const L=(w+.5)/f;b.push(Math.asin(2*L-1))}else if(u==="brdf"){const L=new Float64Array(512);let N=0;for(let V=0;V<512;V++){const W=-Math.PI/2+Math.PI*V/511;L[V]=qt(y,W),N+=L[V]}if(N>0){const V=new Float64Array(512);V[0]=L[0]/N;for(let W=1;W<512;W++)V[W]=V[W-1]+L[W]/N;for(let W=0;W<f;W++){const ct=(W+.5)/f;let ht=0,pt=511;for(;ht<pt;){const q=ht+pt>>1;V[q]<ct?ht=q+1:pt=q}b.push(-Math.PI/2+Math.PI*ht/511)}}}const E=h-v/2,G=h+v/2;for(const w of b){const L=Math.sin(w),N=-Math.cos(w),V=A&&m&&Math.cos(w)>0&&(()=>{const q=h+Math.sin(w)*185/Math.cos(w);return q>=E&&q<=G})();let W=Y,ct=V?.2:.55,ht=1.5;if(R){const q=Xt(w,y),yt=Yt(w),dt=q+yt>0?q/(q+yt):1;({len:W,opacity:ct,width:ht}=Zt(dt,Y,.55,1.5))}const pt=$("line");if(Object.entries({x1:h,y1:S,x2:h+L*W,y2:S+N*W,stroke:"#333","stroke-width":ht,"stroke-dasharray":"6 3",opacity:ct,"marker-end":`url(#${Q}-ah-sample)`}).forEach(([q,yt])=>pt.setAttribute(q,String(yt))),lt.appendChild(pt),V&&!R){const q=h+L*Y,yt=S+N*Y,dt=7;for(const[Gt,Pt,k,J]of[[-dt,-dt,dt,dt],[-dt,dt,dt,-dt]]){const tt=$("line");Object.entries({x1:q+Gt,y1:yt+Pt,x2:q+k,y2:yt+J,stroke:"#e74c3c","stroke-width":2.5,"stroke-linecap":"round"}).forEach(([et,Mt])=>tt.setAttribute(et,String(Mt))),lt.appendChild(tt)}}}}if(U){const b=S-185;if(m){const E=M;for(let G=0;G<E;G++){const w=h-v/2+(G+.5)/E*v,L=b+6,N=w-h,V=S-L,W=Math.atan2(N,V),ct=Math.sqrt(N*N+V*V);let ht=ct,pt=1,q=2;if(R){const k=Yt(W),J=Xt(W,y),tt=k+J>0?k/(k+J):1;({len:ht,opacity:pt,width:q}=Zt(tt,ct,1,2))}const yt=Math.sqrt(N*N+V*V),dt=N/yt,Gt=-V/yt,Pt=$("line");Object.entries({x1:h,y1:S,x2:h+dt*ht,y2:S+Gt*ht,stroke:"#f1c40f","stroke-width":q,opacity:pt,"marker-end":`url(#${Q}-ah-nee)`}).forEach(([k,J])=>Pt.setAttribute(k,String(J))),lt.appendChild(Pt)}}if(o){const E=h+Math.sin(rt)*T,G=S-Math.cos(rt)*T,w=$("line");Object.entries({x1:h,y1:S,x2:E,y2:G,stroke:"#f1c40f","stroke-width":2,"marker-end":`url(#${Q}-ah-nee)`}).forEach(([L,N])=>w.setAttribute(L,String(N))),lt.appendChild(w)}}if(H&&Z&&vt){const b=x!=null?[Math.sin(x),Math.cos(x)]:_,E=pe(y,b);Z.setAttribute("fill",`rgb(${E[0]},${E[1]},${E[2]})`),vt.textContent=`${E[0]},${E[1]},${E[2]}`}}let Ct=null;function Kt(y){const _=P.getBoundingClientRect();return[(y.clientX-_.left)*(n/_.width),(y.clientY-_.top)*(i/_.height)]}function Tt(y,_){const F=y-h,I=S-_;return Math.atan2(F,I)}if(P.addEventListener("pointerdown",y=>{const[_,F]=Kt(y),I=h+Math.sin(rt)*T,C=S-Math.cos(rt)*T,b=Math.sqrt((_-I)**2+(F-C)**2),E=h-Math.sin(it)*z,G=S-Math.cos(it)*z,w=Math.sqrt((_-E)**2+(F-G)**2);D&&b<O+15?Ct="light":w<25?Ct="camera":Ct=D&&b<w?"light":"camera",P.setPointerCapture(y.pointerId),Ct==="light"?rt=Tt(_,F):it=-Tt(_,F),Ft()}),P.addEventListener("pointermove",y=>{if(!Ct)return;const[_,F]=Kt(y);Ct==="light"?rt=Tt(_,F):it=-Tt(_,F),Ft()}),P.addEventListener("pointerup",y=>{Ct=null,P.releasePointerCapture(y.pointerId)}),Ft(),t.appendChild(P),g){const y=document.createElement("div");y.style.cssText="display:flex; align-items:center; justify-content:center; gap:8px; margin-top:4px; font:13px monospace; color:#555;";const _=document.createElement("span");_.textContent="Samples: "+f;const F=document.createElement("input");Object.assign(F,{type:"range",min:3,max:40,value:f,step:1}),F.style.width="180px",F.addEventListener("input",()=>{f=parseInt(F.value),_.textContent="Samples: "+f,Ft()}),y.appendChild(_),y.appendChild(F),t.appendChild(y)}return P}function kn(t,e={}){const n=e.entries||[],i=n.reduce((A,D)=>A+D.flux,0),r=e.barWidth||420,s=36,d=24,c=28,l=d+s+c,a=document.createElementNS.bind(document,"http://www.w3.org/2000/svg"),f=document.createElement("div");f.style.cssText="position:relative; width:fit-content; margin:16px auto;";const u=a("svg");u.setAttribute("width",String(r)),u.setAttribute("height",String(l)),u.setAttribute("viewBox",`0 0 ${r} ${l}`),u.style.display="block";const g={};let o=0;for(const A of n){const D=A.flux/i*r;g[A.type]?g[A.type].x1=o+D:g[A.type]={x0:o,x1:o+D},o+=D}const p={"font-size":"12","font-family":"monospace","text-anchor":"middle"};if(g.emissive){const A=a("text");A.setAttribute("x",String((g.emissive.x0+g.emissive.x1)/2)),A.setAttribute("y",String(d-8)),A.setAttribute("fill","#c87800"),Object.entries(p).forEach(([D,R])=>A.setAttribute(D,R)),A.textContent="Emissive triangles",u.appendChild(A)}if(g.point){const A=a("text");A.setAttribute("x",String((g.point.x0+g.point.x1)/2)),A.setAttribute("y",String(d-8)),A.setAttribute("fill","#2980b9"),Object.entries(p).forEach(([D,R])=>A.setAttribute(D,R)),A.textContent="Point lights",u.appendChild(A)}const x=[];o=0;for(let A=0;A<n.length;A++){const D=n[A],R=D.flux/i*r,K=D.type==="emissive"?"#ffe066":"#b3d9f2",M=D.type==="emissive"?"#f39c12":"#4a9eda",h=a("rect");if(Object.entries({x:o,y:d,width:R,height:s,fill:K,stroke:M,"stroke-width":2}).forEach(([B,Y])=>h.setAttribute(B,String(Y))),u.appendChild(h),R>28){const B=a("text");B.setAttribute("x",String(o+R/2)),B.setAttribute("y",String(d+s/2+4)),B.setAttribute("text-anchor","middle"),B.setAttribute("font-size","11"),B.setAttribute("font-family","monospace"),B.setAttribute("fill","#333"),B.textContent=D.flux.toFixed(2),u.appendChild(B)}const S=a("rect");Object.entries({x:o+1,y:d+1,width:R-2,height:s-2,fill:"none",stroke:"#e74c3c","stroke-width":3,rx:2,visibility:"hidden"}).forEach(([B,Y])=>S.setAttribute(B,String(Y))),u.appendChild(S),x.push({x:o,w:R,entry:D,index:A,highlight:S}),o+=R}const m=a("polygon");m.setAttribute("fill","#e74c3c"),m.setAttribute("visibility","hidden"),u.appendChild(m);const v=a("text");v.setAttribute("x",String(r/2)),v.setAttribute("y",String(d+s+c-4)),v.setAttribute("text-anchor","middle"),v.setAttribute("font-size","12"),v.setAttribute("font-family","monospace"),v.setAttribute("fill","#555"),u.appendChild(v),f.appendChild(u);const H=document.createElement("button");H.textContent="Sample",H.style.cssText=`position:absolute; left:${r+16}px; top:${d}px; height:${s}px; font: 13px monospace; padding: 0 14px; cursor: pointer; border: 2px solid #999; border-radius: 4px; background: #f5f5f5;`,f.appendChild(H);const U=n.filter(A=>A.type==="emissive").length;H.addEventListener("click",()=>{const A=Math.random()*i;let D=0;for(const R of x)if(D+=R.entry.flux,A<D){for(const h of x)h.highlight.setAttribute("visibility","hidden");R.highlight.setAttribute("visibility","visible");const K=A/i*r,M=d-2;m.setAttribute("points",`${K-5},${M-8} ${K+5},${M-8} ${K},${M}`),m.setAttribute("visibility","visible"),R.entry.type==="emissive"?v.textContent="→ sampled emissive triangle "+R.index:v.textContent="→ sampled point light "+(R.index-U);break}}),t.appendChild(f)}function Ln(t){let a=[];function f(){a=[];for(let z=0;z<64;z++)a.push(.05+Math.random()*.95)}f();function u(z){const T=Math.round(z*255),O=Math.round(z*240),it=Math.round(z*40);return`rgb(${T},${O},${it})`}const g=document.createElement("div");g.style.cssText="display:flex; align-items:center; justify-content:center; gap:16px; margin:20px auto;";const o=document.createElement("div");o.style.textAlign="center";const p=document.createElement("div");p.style.cssText="color:#e67e22; font:italic 13px monospace; margin-bottom:6px;",p.textContent="N = 64 lights",o.appendChild(p);const x=document.createElement("div");x.style.cssText="display:grid; grid-template-columns:repeat(8,28px); gap:1px; border:2.5px solid #e67e22; border-radius:6px; padding:2px; background:#fff;";const m=[];for(let z=0;z<64;z++){const T=document.createElement("div");T.style.cssText=`width:28px; height:28px; background:${u(a[z])}; border-radius:2px; transition:outline 0.15s;`,x.appendChild(T),m.push(T)}o.appendChild(x),g.appendChild(o);const v=document.createElement("div");v.style.cssText="text-align:center; font:12px monospace;",v.innerHTML='<div style="color:#4a9eda; font-style:italic; line-height:1.3;">Flux CDF<br>sample</div><div style="font-size:26px; color:#333; margin-top:2px;">→</div>',g.appendChild(v);const H=document.createElement("div");H.style.textAlign="center";const U=document.createElement("div");U.style.cssText="color:#4a9eda; font:italic 13px monospace; margin-bottom:6px;",U.textContent="M = 4 candidates",H.appendChild(U);const A=document.createElement("div");A.style.cssText="display:inline-grid; grid-template-columns:repeat(2,40px); gap:2px; border:2.5px solid #4a9eda; border-radius:8px; padding:3px; background:#fff; box-sizing:content-box;";const D=[];for(let z=0;z<4;z++){const T=document.createElement("div");T.style.cssText="width:40px; height:40px; background:#fff; border:1px solid #ddd; border-radius:3px; transition:all 0.2s;",A.appendChild(T),D.push(T)}H.appendChild(A),g.appendChild(H);const R=document.createElement("div");R.style.cssText="text-align:center; font:12px monospace;",R.innerHTML='<div style="color:#27ae60; font-style:italic; line-height:1.3;">Reservoir<br>selection</div><div style="font-size:26px; color:#333; margin-top:2px;">→</div>',g.appendChild(R);const K=document.createElement("div");K.style.cssText="display:flex; flex-direction:column; align-items:center;";const M=document.createElement("div");M.style.cssText="color:#27ae60; font:italic 13px monospace; margin-bottom:6px;",M.textContent="Final candidate",K.appendChild(M);const h=document.createElement("div");h.style.cssText="width:46px; height:46px; border:3px solid #27ae60; border-radius:8px; background:#fff; transition:background 0.3s;",K.appendChild(h),g.appendChild(K),t.appendChild(g);const S=document.createElement("div");S.style.cssText="text-align:center; font:12px monospace; color:#666; margin:6px auto; min-height:20px; max-width:600px;",t.appendChild(S);const B=document.createElement("button");B.textContent="Sample",B.style.cssText="display:block; margin:8px auto; font:13px monospace; padding:5px 18px; cursor:pointer; border:2px solid #999; border-radius:4px; background:#f5f5f5;",t.appendChild(B);let Y=!1;function nt(){m.forEach((z,T)=>{z.style.outline="none",z.style.background=u(a[T])}),D.forEach(z=>{z.style.background="#fff",z.style.outline="none"}),h.style.background="#fff",S.innerHTML=""}B.addEventListener("click",()=>{if(Y)return;Y=!0,f(),nt(),m.forEach((X,P)=>X.style.background=u(a[P]));const z=a.reduce((X,P)=>X+P,0),T=[];for(let X=0;X<4;X++){let P=Math.random()*z,st=0;for(let Q=0;Q<64;Q++)if(st+=a[Q],P<st){T.push(Q);break}}const O=[];for(let X=0;X<64;X++)O.push(.05+Math.random()*.95);let it=0;function rt(){if(it>=4){setTimeout(()=>$(0,0,-1),500);return}const X=T[it];m[X].style.outline="3px solid #4a9eda",m[X].style.outlineOffset="-1px",D[it].style.background=u(a[X]),S.textContent=`Drawing candidate ${it+1}/4: light #${X}`,it++,setTimeout(rt,500)}function $(X,P,st){if(D.forEach(lt=>lt.style.outline="none"),X>=4){S.innerHTML=`<span style="color:#27ae60">✓ Final: light #${st}</span>  (w_sum = ${P.toFixed(1)})`,Y=!1;return}const Q=T[X],at=a[Q]/z,gt=O[Q],mt=gt/at,bt=P+mt,At=st<0?1:mt/bt,Et=st<0||Math.random()<At,kt=Et?Q:st;D[X].style.outline="3px solid #e74c3c",D[X].style.outlineOffset="-1px",h.style.background=u(a[kt]);const _t=Et?'<span style="color:#27ae60">✓</span>':'<span style="color:#e74c3c">✗</span>';S.innerHTML=`w = p̂/q = ${gt.toFixed(2)}/${at.toFixed(3)} = ${mt.toFixed(1)}  P(accept) = ${At.toFixed(2)} ${_t}`,setTimeout(()=>$(X+1,bt,kt),500*1.5)}setTimeout(rt,200)})}function $e(t){const e=new ArrayBuffer(t.nodeCount*32),n=new Float32Array(e),i=new Uint32Array(e);for(let r=0;r<t.nodeCount;r++){const s=r*8;n[s]=t.data[s],n[s+1]=t.data[s+1],n[s+2]=t.data[s+2],i[s+3]=t.data[s+3],n[s+4]=t.data[s+4],n[s+5]=t.data[s+5],n[s+6]=t.data[s+6],i[s+7]=t.data[s+7]}return e}function Ne(t){let e=0;const n=[{idx:0,depth:0}];for(;n.length>0;){const{idx:i,depth:r}=n.pop();r>e&&(e=r);const s=i*8;t[s+3]===0&&(n.push({idx:i+1,depth:r+1}),n.push({idx:t[s+7],depth:r+1}))}return e}function He(t,e,n,i){const r=t/e;return Math.round(n*Math.pow(i/n,r))}function Ve(t,e,n,i){return Math.round(e*Math.log(t/n)/Math.log(i/n))}const Nt={roughness:.3,metalness:0,fresnel:[.04,.04,.04],baseColor:[1,.71,.29]};$t(document.getElementById("viz-raytracer"),{...Nt,incomingAngle:55,lightAngle:30,areaLight:!0});$t(document.getElementById("viz-naive"),{...Nt,incomingAngle:55,lightAngle:30,showLightRay:!1,swatchLightAngle:0,showSamples:!0,sampling:"uniform",numSamples:9,areaLight:!0});$t(document.getElementById("viz-nee-direct"),{...Nt,incomingAngle:55,lightAngle:30,showSwatch:!1,showLightRay:!1,showNeeRays:!0,areaLight:!0,areaLightWidth:140,draggableLight:!1,width:340});$t(document.getElementById("viz-nee-indirect"),{...Nt,incomingAngle:55,lightAngle:30,showSwatch:!1,showLightRay:!1,showSamples:!0,sampling:"uniform",numSamples:9,crossOutAreaHits:!0,areaLight:!0,areaLightWidth:140,draggableLight:!1,width:340});$t(document.getElementById("viz-brdfis"),{...Nt,metalness:.1,incomingAngle:55,lightAngle:30,showLobe:!0,showSwatch:!1,showLightRay:!1,showSamples:!0,sampling:"brdf",numSamples:9,areaLight:!0,showSlider:!0});$t(document.getElementById("viz-mis-naive"),{...Nt,metalness:1,baseColor:[.8,.8,.8],fresnel:[.95,.95,.95],roughness:.15,incomingAngle:55,lightAngle:30,showLobe:!0,showSwatch:!1,showLightRay:!1,showSamples:!0,sampling:"brdf",numSamples:9,crossOutAreaHits:!0,showNeeRays:!0,areaLight:!0,areaLightWidth:140,draggableLight:!1,showSlider:!0});$t(document.getElementById("viz-mis-balanced"),{...Nt,metalness:0,baseColor:[.8,.8,.8],fresnel:[.04,.04,.04],roughness:.1,incomingAngle:55,lightAngle:30,showLobe:!0,showSwatch:!1,showLightRay:!1,showSamples:!0,sampling:"brdf",numSamples:9,showNeeRays:!0,areaLight:!0,areaLightWidth:140,draggableLight:!1,showSlider:!0,misWeights:!0,misWeightMode:"length",neeRayCount:1});kn(document.getElementById("viz-nee-cdf"),{entries:[{flux:.2,type:"emissive"},{flux:.2,type:"emissive"},{flux:.38,type:"point"}]});Ln(document.getElementById("viz-ris-reservoir"));async function Dn(){const t=document.getElementById("canvas");t.height=Math.floor(window.innerHeight/8)*8;const{device:e,context:n,format:i,hasTimestamps:r}=await wn(t),s=Je(t),d=Fn(e,i,Qe),c=Re(e,Ee+Ze),l=Re(e,Ee+Ke),a=Cn(e),f=ze(e,t.width,t.height),u=ze(e,t.width,t.height),g=new Uint8Array(t.width*t.height*16),o=En(e,d,f,a);let p=0;function x(){p=0,e.queue.writeBuffer(f,0,g),e.queue.writeBuffer(u,0,g)}let m=null,v=null,H={buffer:new ArrayBuffer(0),count:0},U=null,A=null,D=null,R=0,K=!1,M=null;const h=document.getElementById("loading-overlay");function S(k){return h.textContent=k,h.style.display="",new Promise(J=>setTimeout(J,0))}function B(){h.style.display="none"}const Y=document.getElementById("scene-cornell"),nt=document.getElementById("scene-sponza"),z=document.getElementById("dragon-res"),T=document.getElementById("light-count"),O=document.getElementById("light-count-val"),it=document.getElementById("light-count-apply");let rt=100;z.addEventListener("change",async()=>{await st("cornell"),_()});const $=document.getElementById("light-emission"),X=document.getElementById("light-emission-val");T.addEventListener("input",()=>{O.textContent=T.value}),it.addEventListener("click",async()=>{rt=parseInt(T.value),await st("sponza"),_()}),$.addEventListener("input",()=>{const k=parseFloat($.value);X.textContent=k.toFixed(1),v&&M==="sponza"&&(v.materials[1].emission=k,x(),_())});function P(k){Y.style.display=k==="cornell"?"":"none",nt.style.display=k==="sponza"?"":"none"}async function st(k){K=!0,M=k,P(k),console.log(`Loading scene: ${k}...`);const J=performance.now(),tt=Pe[k]||Pe.cornell;s.position=[...tt.position],s.target=[...tt.target];const et=k==="sponza"?{lightCount:rt}:{dragonRes:z.value};await S("Loading mesh..."),m=await gn(k,et),v=bn(m);const Mt=v.meshes.length,St=v.instances.length;await S(`Building BLAS (${Mt} unique mesh${Mt>1?"es":""}, ${St} instances)...`);const Ut=v.meshes.map(j=>sn({positions:j.positions,indices:j.indices})),me=[],ne=[],we=[];let ie=0,Qt=0,ge=0;for(let j=0;j<Mt;j++)me.push(ie),ne.push(ge),we.push(Qt),ie+=Ut[j].nodeCount,Qt+=v.meshes[j].positions.length/3,ge+=v.meshes[j].indices.length/3;const _e=new Float32Array(Qt*3),Ae=new Float32Array(Qt*3),Ce=new Float32Array(Qt*2),Jt=new Uint32Array(ge*3);for(let j=0;j<Mt;j++){const ot=v.meshes[j],Bt=Ut[j],ut=we[j],ft=ne[j];_e.set(ot.positions,ut*3),Ae.set(ot.normals,ut*3),Ce.set(ot.uvs,ut*2);for(let wt=0;wt<Bt.primitives.length;wt++){const It=Bt.primitives[wt]*3;Jt[(ft+wt)*3]=ot.indices[It]+ut,Jt[(ft+wt)*3+1]=ot.indices[It+1]+ut,Jt[(ft+wt)*3+2]=ot.indices[It+2]+ut}}H=xn(v,Jt,_e,ne);const We=v.instances.map(j=>{const ot=Ut[j.meshIndex].data,[Bt,ut,ft,wt,It,oe]=[ot[0],ot[1],ot[2],ot[4],ot[5],ot[6]],Ye=[[Bt,ut,ft],[wt,ut,ft],[Bt,It,ft],[wt,It,ft],[Bt,ut,oe],[wt,ut,oe],[Bt,It,oe],[wt,It,oe]],xt=j.transform,zt=new Float32Array([1e30,1e30,1e30]),Rt=new Float32Array([-1e30,-1e30,-1e30]);for(const[ye,xe,be]of Ye){const ae=xt[0]*ye+xt[4]*xe+xt[8]*be+xt[12],le=xt[1]*ye+xt[5]*xe+xt[9]*be+xt[13],ce=xt[2]*ye+xt[6]*xe+xt[10]*be+xt[14];ae<zt[0]&&(zt[0]=ae),ae>Rt[0]&&(Rt[0]=ae),le<zt[1]&&(zt[1]=le),le>Rt[1]&&(Rt[1]=le),ce<zt[2]&&(zt[2]=ce),ce>Rt[2]&&(Rt[2]=ce)}return{minCorner:zt,maxCorner:Rt}});await S(`Building TLAS (${St} instances)...`);const ve=rn(We),Me=Ne(ve.data),Fe=Math.max(...Ut.map(j=>Ne(j.data)));R=Me+Fe+1,V.max=String(R),console.log(`BVH build: ${(performance.now()-J).toFixed(1)}ms | TLAS: ${ve.nodeCount} nodes (depth ${Me}) | BLASes: ${ie} nodes total (max depth ${Fe}) | ${Mt} unique meshes, ${St} instances`);const Xe=$e(ve),Se=new ArrayBuffer(ie*32);for(let j=0;j<Ut.length;j++)new Uint8Array(Se,me[j]*32).set(new Uint8Array($e(Ut[j])));const Be=36,se=new Float32Array(St*Be),re=new Uint32Array(se.buffer);for(let j=0;j<St;j++){const ot=v.instances[j],Bt=fn(ot.transform),ut=j*Be;for(let ft=0;ft<16;ft++)se[ut+ft]=ot.transform[ft];for(let ft=0;ft<16;ft++)se[ut+16+ft]=Bt[ft];re[ut+32]=me[ot.meshIndex],re[ut+33]=ne[ot.meshIndex],re[ut+34]=ot.materialId,re[ut+35]=0}await S("Uploading to GPU..."),Bn(U),U=Sn(e,{positions:_e,normals:Ae,uvs:Ce,indices:Jt},Xe,Se,se.buffer,H.buffer),A=Oe(e,c,a,U,f,u),D=Oe(e,l,a,U,f,u),x(),K=!1,B(),console.log(`Scene "${k}" ready (${(performance.now()-J).toFixed(0)}ms)`)}const Q=document.getElementById("stats"),at=[],gt=[],mt=1e3,bt=An(e,r);bt&&(bt.onGpuTime=k=>{gt.push(k),gt.length>mt&&gt.shift(),Et()});let At=0;function Et(){const k=at.reduce((tt,et)=>tt+et,0)/(at.length||1);let J=`CPU  last: ${At.toFixed(2)}ms  avg(${at.length}): ${k.toFixed(2)}ms`;if(bt){const tt=gt.reduce((Mt,St)=>Mt+St,0)/(gt.length||1),et=gt.at(-1)??0;J+=`
GPU  last: ${et.toFixed(2)}ms  avg(${gt.length}): ${tt.toFixed(2)}ms`}Q.textContent=J}const kt={position:[200,400,200],color:[1,1,1],intensity:.03,direction:[0,-1,0],angle:360},_t=document.getElementById("controls-toggle"),lt=document.getElementById("controls-body");_t.addEventListener("click",()=>{const k=lt.style.display!=="none";lt.style.display=k?"none":"",_t.textContent=k?"CONTROLS ▸":"CONTROLS ▾"});let Lt=!0;const jt=document.getElementById("point-light");jt.addEventListener("change",()=>{Lt=jt.checked,x(),_()});let Ht=!1,Z=0;const vt=document.getElementById("render-mode");vt.addEventListener("change",()=>{Ht=vt.checked,x(),_()});const Vt=document.getElementById("nee-mode"),Dt=document.getElementById("brdfis-mode");Vt.addEventListener("change",()=>{Z=Vt.checked?Z|1:Z&-2,x(),_()}),Dt.addEventListener("change",()=>{Z=Dt.checked?Z|2:Z&-3,x(),_()});const Wt=document.getElementById("mis-mode");Wt.addEventListener("change",()=>{Z=Wt.checked?Z|4:Z&-5,x(),_()});const ee=document.getElementById("ris-mode"),qt=document.getElementById("ris-samples"),pe=document.getElementById("ris-samples-val"),Xt=1,Yt=256,Zt=1e3;let Ft=32,Ct=2;const Kt=document.getElementById("ris-target");qt.value=String(Ve(Ft,Zt,Xt,Yt)),ee.addEventListener("change",()=>{Z=ee.checked?Z|8:Z&-9,x(),_()});const Tt=document.getElementById("restir-mode");Tt.addEventListener("change",()=>{Z=Tt.checked?Z|16:Z&-17,x(),_()}),qt.addEventListener("input",()=>{Ft=He(parseInt(qt.value),Zt,Xt,Yt),pe.textContent=String(Ft),x(),_()}),Kt.addEventListener("change",()=>{Ct=parseInt(Kt.value),x(),_()});const y=document.getElementById("scene-select");y.addEventListener("change",async()=>{await st(y.value),_()});function _(){if(K||!v||!m)return;const k=performance.now(),J=Ht?l:c,tt=Ht?D:A;if(!tt)return;const et=Lt&&M==="cornell"?[...m.lights,kt]:m.lights,St=(Z&8)!==0?Ft:1;e.queue.writeBuffer(a,0,Mn(s,et,v.materials,p,w,L,N,t.width,t.height,Z,H.count,St,Ct)),In(e,n,d,J,o,tt,t.width,t.height,bt),p++,At=performance.now()-k,at.push(At),at.length>mt&&at.shift(),Et()}const F=1,I=4096,C=1e3;let b=32;const E=document.getElementById("spp-slider"),G=document.getElementById("spp-val");E.value=String(Ve(b,C,F,I)),E.addEventListener("input",()=>{const k=He(parseInt(E.value),C,F,I);k<b&&x(),b=k,G.textContent=String(b),p<b&&_()});let w=-1,L=4,N=1;const V=document.getElementById("bvh-depth"),W=document.getElementById("bvh-depth-val"),ct=document.getElementById("bvh-heat"),ht=document.getElementById("bvh-heat-val");V.addEventListener("input",()=>{w=parseInt(V.value),W.textContent=w<0?"off":String(w),x(),_()}),ct.addEventListener("input",()=>{L=parseInt(ct.value),ht.textContent=String(L),x(),_()});const pt=document.getElementById("bvh-early-stop");pt.addEventListener("change",()=>{N=pt.checked?1:0,x(),_()}),await st(y.value),_();const q=new Set,yt=200,dt=1.2;window.addEventListener("keydown",k=>{q.add(k.code),k.code.startsWith("Arrow")&&k.preventDefault()}),window.addEventListener("keyup",k=>q.delete(k.code));let Gt=performance.now();function Pt(k){const J=Math.min((k-Gt)/1e3,.1);Gt=k;const tt=yt*J;let et=!1;q.has("KeyW")&&(Ie(s,tt),et=!0),q.has("KeyS")&&(Ie(s,-tt),et=!0),q.has("KeyA")&&(ue(s,-tt,0),et=!0),q.has("KeyD")&&(ue(s,tt,0),et=!0),q.has("KeyR")&&(ue(s,0,tt),et=!0),q.has("KeyF")&&(ue(s,0,-tt),et=!0),q.has("ArrowLeft")&&(ke(s,dt*J),et=!0),q.has("ArrowRight")&&(ke(s,-dt*J),et=!0),q.has("ArrowUp")&&(Le(s,dt*J),et=!0),q.has("ArrowDown")&&(Le(s,-dt*J),et=!0),et&&x(),p<b&&_(),requestAnimationFrame(Pt)}requestAnimationFrame(Pt)}Dn().catch(t=>console.error(t));
