// === raytracer.wgsl ===
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
      let lightVec = u.lights[i].position - sp.world_pos;
      let lightDir = normalize(lightVec);

      var shadow_ray: Ray;
      shadow_ray.origin    = sp.world_pos + sp.normal * 0.1;
      shadow_ray.direction = lightDir;

      var shadow_hit: Hit;
      if (!rayTrace(shadow_ray, &shadow_hit, true, length(lightVec))) {
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
