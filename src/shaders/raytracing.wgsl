struct VertexOutput {
  @builtin(position) clip_pos: vec4f,
  @location(0) screen_pos: vec2f,
};

struct PointLight{
  position: vec3<f32>,
  _pad0: f32, // will be intensity later
  color: vec3<f32>,
  _pad1: f32,
  // will be an other padding later
}

struct Material{
  baseColor: vec3<f32>,
  _pad0: f32
}

struct RayUniforms {
  camera_pos: vec3f,
  fov_factor: f32,          // uses camera_pos padding slot
  camera_forward: vec3f,
  aspect_ratio: f32,        // uses camera_forward padding slot
  camera_right: vec3f,
  _pad0: f32,               // unused padding
  camera_up: vec3f,
  _pad1: f32,               // unused padding
  nbLights: f32,
  _pad2: f32,
  _pad3: f32,
  _pad4: f32,
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
    let color = material.baseColor;

    let n0 = vec3f(normals[i0 * 3 + 0], normals[i0 * 3 + 1], normals[i0 * 3 + 2]);
    let n1 = vec3f(normals[i1 * 3 + 0], normals[i1 * 3 + 1], normals[i1 * 3 + 2]);
    let n2 = vec3f(normals[i2 * 3 + 0], normals[i2 * 3 + 1], normals[i2 * 3 + 2]);

    let p0 = vec3f(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    let p1 = vec3f(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    let p2 = vec3f(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);


    let bary = hit_data.barycentricCoords;
    let normal = normalize(bary.x * n0 + bary.y * n1 + bary.z * n2);
    let world_pos = bary.x * p0 + bary.y * p1 + bary.z * p2;



    for(var i = 0u; i < u32(uniforms.nbLights); i++){
      let lightDir = normalize(uniforms.lights[i].position - world_pos);
      let lambertFactor = max(0.0, dot(lightDir, normalize(normal)));
      let lightColor = uniforms.lights[i].color;


      var shadow_ray: Ray;

      shadow_ray.direction = lightDir;
      shadow_ray.origin = world_pos + normal * 0.001;

      let is_shadow = rayTrace(shadow_ray, &hit_data);


      if(is_shadow && hit_data.t < length(uniforms.lights[i].position - world_pos)){ // we don't use hit data here so it's jsut a placeholder
        // factor is a parameter, lower = stronger shadow
        output_color += (color * lambertFactor * lightColor * 0.3);
      }
      else{
        output_color += (color * lambertFactor * lightColor);
      }
    }
    output_color /= uniforms.nbLights;
  }

  return vec4f(output_color, 1.0);
}
