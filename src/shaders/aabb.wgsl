// Only needs the MVP from the full uniform buffer
struct Uniforms {
  mvp: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform>       uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> vertices : array<f32>;


// ============================================================================
// AABB wireframe
// ============================================================================

@vertex
fn aabbVertexMain(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
  let pos = vec3f(vertices[vid*3u], vertices[vid*3u+1u], vertices[vid*3u+2u]);
  return uniforms.mvp * vec4f(pos, 1.0);
}

@fragment
fn aabbFragmentMain() -> @location(0) vec4f {
  return vec4f(1.0, 1.0, 0.0, 1.0);
}
