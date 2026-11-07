import"./modulepreload-polyfill-B5Qt9EMX.js";/* empty css              */import{i as P}from"./webgpu-D3-56OEV.js";import{i as C,g as G}from"./camera-Dvgm1X7S.js";const V=`struct Uniforms {
  mvp : mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;


struct VertexInput {
  @location(0) pos: vec3f,
  @location(1) color: vec3f,
  @builtin(instance_index) instance: u32
};


struct VertexOutput {
  @builtin(position) pos : vec4f,
  @location(0) color: vec4f
};

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.pos = uniforms.mvp * vec4f(input.pos, 1);
  output.color = vec4f(input.color, 1);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return input.color;
}
`,o=document.getElementById("canvas"),i=document.getElementById("warning-container");function x(){i.innerHTML=`
    <div class="webgpu-warning">
      <p><strong>⚠ WebGPU is not supported or not enabled in your browser</strong></p>
      <p><strong>Chrome:</strong></p>
      <ul>
        <li>Go to <code>chrome://flags</code></li>
        <li>Search for "Unsafe WebGPU"</li>
        <li>Enable the flag and restart Chrome</li>
        <li>Check that Vulkan is enabled at <code>chrome://gpu</code></li>
      </ul>
      <p><strong>Firefox:</strong></p>
      <ul>
        <li>Go to <code>about:config</code></li>
        <li>Search for <code>dom.webgpu.enabled</code></li>
        <li>Set it to true and restart Firefox</li>
        <li><em>Note: Firefox on Linux does not support Vulkan, so WebGPU will be much slower</em></li>
      </ul>
    </div>`,o.style.display="none"}function k(){i.innerHTML=`
    <div style="text-align:center;margin:0 auto 1rem;max-width:600px;">
      <p id="vulkan-toggle" style="color:red;font-size:0.9rem;cursor:pointer;border:1px solid red;background:#ffe6e6;padding:0.5rem;margin:0;">
        Running too slow? Click here.
      </p>
    </div>`,document.getElementById("vulkan-toggle").addEventListener("click",()=>{i.innerHTML=`
      <div style="text-align:center;margin:0 auto 1rem;max-width:600px;">
        <div style="border:1px solid red;background:#ffe6e6;padding:1rem;text-align:left;">
          <p><strong>⚠ Low performance detected</strong></p>
          <p>Your setup may be too slow for complex renderings.</p>
          <p><strong>Chrome:</strong></p>
          <ul>
            <li>Go to <code>chrome://gpu</code> to check your graphics feature status</li>
            <li>Make sure hardware acceleration is enabled in Chrome settings</li>
            <li>Update your graphics drivers if Vulkan/D3D12 is disabled</li>
          </ul>
          <p><strong>Firefox on Linux:</strong></p>
          <ul>
            <li>Firefox does not support Vulkan on Linux. Too bad :(</li>
            <li>Consider using Chrome for better WebGPU performance</li>
          </ul>
          <button onclick="this.closest('div').parentElement.remove()">Close</button>
        </div>
      </div>`})}async function L(){if(!navigator.gpu){x();return}let a;try{const{device:e,context:v,_adapter:T}=await P(o),h=C(o),w=G(h),B=navigator.gpu.getPreferredCanvasFormat(),s=new Float32Array([-1,-1,0,1,-1,0,0,1,0]),n=new Uint32Array([0,1,2]),u=new Float32Array([1,0,0,0,1,0,0,0,1]),l=e.createBuffer({label:"Vertices",size:s.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),c=e.createBuffer({label:"Vertex indices",size:n.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),f=e.createBuffer({label:"Vertex colors",size:u.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(l,0,s),e.queue.writeBuffer(c,0,n),e.queue.writeBuffer(f,0,u);const d=e.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(d,0,w);const p=e.createShaderModule({label:"Cell shader",code:V}),g=e.createRenderPipeline({label:"Cell pipeline",layout:"auto",vertex:{module:p,entryPoint:"vertexMain",buffers:[{arrayStride:12,attributes:[{format:"float32x3",offset:0,shaderLocation:0}]},{arrayStride:12,attributes:[{format:"float32x3",offset:0,shaderLocation:1}]}]},fragment:{module:p,entryPoint:"fragmentMain",targets:[{format:B}]}}),U=e.createBindGroup({layout:g.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:d}}]}),m=()=>{const r=e.createCommandEncoder(),t=r.beginRenderPass({colorAttachments:[{view:v.getCurrentTexture().createView(),loadOp:"clear",clearValue:{r:0,g:0,b:.4,a:1},storeOp:"store"}]});t.setPipeline(g),t.setBindGroup(0,U),t.setVertexBuffer(0,l),t.setVertexBuffer(1,f),t.setIndexBuffer(c,"uint32"),t.drawIndexed(n.length,1),t.end(),e.queue.submit([r.finish()])};m(),a=window.setTimeout(()=>{let r=0;const t=performance.now(),y=2e3,b=()=>{m(),r++,performance.now()-t<y?requestAnimationFrame(b):r/(performance.now()-t)*1e3<10&&k()};requestAnimationFrame(b)},500)}catch(e){console.error("WebGPU initialization failed:",e),x(),clearTimeout(a)}}L();
