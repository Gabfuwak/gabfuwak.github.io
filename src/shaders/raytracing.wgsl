struct VertexOutput {
  @builtin(position) clip_pos: vec4f,
  @location(0) screen_pos: vec2f,
};

@vertex
fn vertexMain(@location(0) pos: vec2f) -> VertexOutput {
  var output: VertexOutput;
  output.clip_pos = vec4f(pos, 0.0, 1.0);
  output.screen_pos = pos;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return vec4f(1.0, 0.0, 0.0, 1.0); // red
}
