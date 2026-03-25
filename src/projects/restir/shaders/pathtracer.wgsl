// === pathtracer.wgsl ===
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
  let light    = u.lights[chosen];
  let lightVec = light.position - sp.world_pos;
  let lightDir = normalize(lightVec);
  var shadow_ray: Ray;
  shadow_ray.origin    = sp.world_pos + sp.normal * 0.001;
  shadow_ray.direction = lightDir;
  var shadow_hit: Hit;
  if (!rayTrace(shadow_ray, &shadow_hit, true, length(lightVec))) {
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
  let lightDir    = normalize(lightVec);
  let dist_m      = length(lightVec) / 1000.0; // mm -> m
  let cos_light   = max(0.0, dot(ls.normal, -lightDir));
  let cos_surface = max(0.0, dot(sp.normal, lightDir));
  if (cos_light <= 0.0 || cos_surface <= 0.0) { return vec3f(0.0); }

  var shadow_ray: Ray;
  shadow_ray.origin    = sp.world_pos + sp.normal * 0.001;
  shadow_ray.direction = lightDir;
  var shadow_hit: Hit;
  if (!rayTrace(shadow_ray, &shadow_hit, true, length(lightVec) - 0.5)) {
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
          let lightDir = normalize(lightVec);
          let dist_m = length(lightVec) / 1000.0;
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
              let visible = !rayTrace(shadow_ray, &shadow_hit, true, length(lightVec) - 0.5);
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
          let lightDir = normalize(lightVec);
          let cos_surface = max(0.0, dot(sp.normal, lightDir));
          let cos_light = max(0.0, dot(ls.normal, -lightDir));
          if (cos_surface > 0.0 && cos_light > 0.0) {
            var shadow_ray: Ray;
            shadow_ray.origin = sp.world_pos + sp.normal * 0.001;
            shadow_ray.direction = lightDir;
            var shadow_hit: Hit;
            if (!rayTrace(shadow_ray, &shadow_hit, true, length(lightVec) - 0.5)) {
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
    if (all(throughput < vec3f(f32(rr_seed) / f32(0xFFFFFFFFu)))) { break; }
  }

  // Touch unused bindings so auto layout keeps them alive
  let _uv = uvs[0];
  let _et = emissive_tris[0];
  let _rv = reservoir[0];

  color = applyBvhOverlay(screen_pos, color);

  let idx = px.y * u32(u.screen_width) + px.x;
  accum[idx] = accum[idx] + vec4f(color, 1.0);
}
