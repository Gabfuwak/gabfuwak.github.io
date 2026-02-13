fn pcg_hash(seed: u32) -> u32 {
  var state = seed * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn random(t: vec2<f32>) -> f32 {
  let seed = pcg_hash(bitcast<u32>(t.x) ^ pcg_hash(bitcast<u32>(t.y)));
  return f32(seed) / f32(0xFFFFFFFFu);
}


fn valueNoise2D(x: vec2<f32>, freq: f32, amp: f32) -> f32{
  let pos_abs = x * freq;
  let pos_grid = floor(pos_abs);
  let pos_rel = pos_abs - pos_grid;

  let v00 = random(pos_grid);
  let v10 = random(pos_grid + vec2f(1.0, 0.0));
  let v01 = random(pos_grid + vec2f(0.0, 1.0));
  let v11 = random(pos_grid + vec2f(1.0, 1.0));

  let t = smoothstep(vec2f(0.0), vec2f(1.0), pos_rel);

  // interpolate both values on y axis
  let value_0x = mix(v00, v01, t.y);
  let value_1x = mix(v10, v11, t.y);

  // return interpolation on x axis
  return mix(value_0x, value_1x, t.x) * amp;
}

fn gradient2D(p: vec2f) -> vec2f {
  // sample from circle to get rid of noise
  let phi = random(p) * 6.28318530718;
  return vec2f(cos(phi), sin(phi));
}

// let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
fn perlinNoise2D(x: vec2f, freq: f32, amp: f32) -> f32 {
  let pos_abs = x * freq;
  let pos_grid = floor(pos_abs);
  let pos_rel = pos_abs - pos_grid;

  let g00 = gradient2D(pos_grid);
  let g10 = gradient2D(pos_grid + vec2f(1.0, 0.0));
  let g01 = gradient2D(pos_grid + vec2f(0.0, 1.0));
  let g11 = gradient2D(pos_grid + vec2f(1.0, 1.0));

  let v00 = dot(g00, pos_rel);
  let v10 = dot(g10, pos_rel - vec2f(1.0, 0.0));
  let v01 = dot(g01, pos_rel - vec2f(0.0, 1.0));
  let v11 = dot(g11, pos_rel - vec2f(1.0, 1.0));

  let t = pos_rel * pos_rel * pos_rel * (pos_rel * (pos_rel * 6.0 - 15.0) + 10.0);

  // interpolate both values on y axis
  let value_0x = mix(v00, v01, t.y);
  let value_1x = mix(v10, v11, t.y);

  // return interpolation on x axis
  return mix(value_0x, value_1x, t.x) * amp;
}

fn octavePerlin2D(x: vec2f, freq: f32, amp: f32, octaves: u32) -> f32 {
  var ret : f32 = 0.0;
  for(var i = 1u; i <= octaves; i++){
    ret += perlinNoise2D(x, freq*pow(f32(i), 2), amp/pow(f32(i), 2));
  }
  return ret;
}


fn octaveValue2D(x: vec2f, freq: f32, amp: f32, octaves: u32) -> f32 {
  var ret : f32 = 0.0;
  for(var i = 1u; i <= octaves; i++){
    ret += valueNoise2D(x, freq*pow(f32(i), 2), amp/pow(f32(i), 2));
  }
  return ret;
}
