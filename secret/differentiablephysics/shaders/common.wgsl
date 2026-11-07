struct PointLight {
  position:  vec3f,
  intensity: f32,
  color:     vec3f,
  _pad1:     f32,
};

struct Material {
  baseColor: vec3f,
  roughness: f32,
  fresnel:   vec3f,
  metalness: f32,
};

struct Uniforms {
  camera_pos:     vec3f,
  fov_factor:     f32,
  camera_forward: vec3f,
  aspect_ratio:   f32,
  camera_right:   vec3f,
  nb_lights:      f32,
  camera_up:      vec3f,
  _pad_camera:    f32,
  lights:         array<PointLight, 4>,
  screen_width:   f32,
  screen_height:  f32,
  _pad_screen:    vec2f,
  materials:      array<Material, 16>,
  floor_height:   f32,
  floor_material: u32,
  visible:        u32,
  _pad_display:   u32,
};

struct Sphere {
  center:     vec3f,
  radius:     f32,
  materialId: u32,
  _pad1:      u32,
  _pad2:      u32,
  _pad3:      u32,
};

struct Triangle {
  a:          u32,
  b:          u32,
  c:          u32,
  materialId: u32,
};

const HIT_SPHERE = 0u;
const HIT_TRIANGLE = 1u;
const HIT_FLOOR = 2u;

const SHOW_PARTICLES = 1u;
const SHOW_TRIANGLES = 2u;
const SHOW_FLOOR = 4u;

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(8) var<storage, read_write> accum: array<vec4f>;

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

  let lightDistance = length(lightVec);
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

struct Hit {
  kind:  u32,
  index: u32,
  t:     f32,
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

fn raySphereHit(ray: Ray, sphere: Sphere) -> f32 {
  let oc = ray.origin - sphere.center;
  let a = dot(ray.direction, ray.direction);
  let half_b = dot(oc, ray.direction);
  let c = dot(oc, oc) - sphere.radius * sphere.radius;
  let discriminant = half_b * half_b - a * c;
  if (discriminant < 0.0) { return -1.0; }

  let root = sqrt(discriminant);
  let near_t = (-half_b - root) / a;
  if (near_t > 0.00000001) { return near_t; }
  let far_t = (-half_b + root) / a;
  if (far_t > 0.00000001) { return far_t; }
  return -1.0;
}

// Double-sided Moller-Trumbore intersection; vertices follow the particle buffer.
fn rayTriangleHit(ray: Ray, triangle: Triangle) -> f32 {
  let a = spheres[triangle.a].center;
  let edge1 = spheres[triangle.b].center - a;
  let edge2 = spheres[triangle.c].center - a;
  let h = cross(ray.direction, edge2);
  let determinant = dot(edge1, h);
  if (abs(determinant) < 1e-10) { return -1.0; }

  let inverse = 1.0 / determinant;
  let s = ray.origin - a;
  let b1 = inverse * dot(s, h);
  if (b1 < 0.0 || b1 > 1.0) { return -1.0; }
  let q = cross(s, edge1);
  let b2 = inverse * dot(ray.direction, q);
  if (b2 < 0.0 || b1 + b2 > 1.0) { return -1.0; }
  let t = inverse * dot(edge2, q);
  if (t > 0.00000001) { return t; }
  return -1.0;
}

fn rayFloorHit(ray: Ray) -> f32 {
  if (abs(ray.direction.y) < 1e-8) { return -1.0; }
  let t = (u.floor_height - ray.origin.y) / ray.direction.y;
  if (t > 0.00000001) { return t; }
  return -1.0;
}

fn rayTrace(ray: Ray, hit: ptr<function, Hit>, is_shadow: bool, max_t: f32) -> bool {
  var closest_t = max_t;
  var found_hit = false;

  if ((u.visible & SHOW_PARTICLES) != 0u) {
    for (var i = 0u; i < arrayLength(&spheres); i++) {
      let candidate_t = raySphereHit(ray, spheres[i]);
      if (candidate_t > 0.0 && candidate_t < closest_t) {
        closest_t = candidate_t;
        (*hit).kind = HIT_SPHERE;
        (*hit).index = i;
        (*hit).t = candidate_t;
        found_hit = true;
        if (is_shadow) { return true; }
      }
    }
  }
  if ((u.visible & SHOW_TRIANGLES) != 0u) {
    for (var i = 0u; i < arrayLength(&triangles); i++) {
      let candidate_t = rayTriangleHit(ray, triangles[i]);
      if (candidate_t > 0.0 && candidate_t < closest_t) {
        closest_t = candidate_t;
        (*hit).kind = HIT_TRIANGLE;
        (*hit).index = i;
        (*hit).t = candidate_t;
        found_hit = true;
        if (is_shadow) { return true; }
      }
    }
  }
  if ((u.visible & SHOW_FLOOR) != 0u) {
    let candidate_t = rayFloorHit(ray);
    if (candidate_t > 0.0 && candidate_t < closest_t) {
      (*hit).kind = HIT_FLOOR;
      (*hit).index = 0u;
      (*hit).t = candidate_t;
      found_hit = true;
    }
  }
  return found_hit;
}

fn resolve_hit(ray: Ray, hit: Hit) -> SurfacePoint {
  var sp: SurfacePoint;
  sp.world_pos = ray.origin + hit.t * ray.direction;
  if (hit.kind == HIT_SPHERE) {
    let sphere = spheres[hit.index];
    sp.normal = normalize(sp.world_pos - sphere.center);
    sp.material = u.materials[sphere.materialId];
  } else if (hit.kind == HIT_TRIANGLE) {
    let triangle = triangles[hit.index];
    let a = spheres[triangle.a].center;
    let b = spheres[triangle.b].center;
    let c = spheres[triangle.c].center;
    sp.normal = normalize(cross(b - a, c - a));
    if (dot(sp.normal, ray.direction) > 0.0) { sp.normal = -sp.normal; }
    sp.material = u.materials[triangle.materialId];
  } else {
    sp.normal = vec3f(0.0, 1.0, 0.0);
    if (ray.direction.y > 0.0) { sp.normal = -sp.normal; }
    sp.material = u.materials[u.floor_material];
  }
  return sp;
}
