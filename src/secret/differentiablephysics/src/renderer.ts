import { getCameraBasis } from "./camera";
import type { Camera } from "./camera";
import type { Material, PointLight, Particle, Triangle, RenderOptions } from "./scene";
import { floor } from "./scene";

export async function initWebGPU(canvas: HTMLCanvasElement) {
  if (!navigator.gpu) throw new Error("WebGPU not supported");

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("No GPU adapter found");

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu")!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  return { device, context, format };
}

export function createAccumulationBuffer(device: GPUDevice, screenWidth: number, screenHeight: number){
  return device.createBuffer({
    size: screenWidth*screenHeight*16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })

}

// Uniform layout (must match common.wgsl and display.wgsl Uniforms):
// Camera:         4 × vec4f =  64 bytes (floats 0–15)
// Lights[4]:      4 × 8f    = 128 bytes (floats 16–47)
// Screen+pad:     1 × vec4f =  16 bytes (floats 48–51)
// Materials[16]: 16 × 8f    = 512 bytes (floats 52–179)
// Floor+display:  1 × vec4  =  16 bytes (words 180–183)
// Total: 736 bytes = 184 words.
const UNIFORM_FLOATS = 184;

export function createUniformBuffer(device: GPUDevice) {
  return device.createBuffer({
    size: UNIFORM_FLOATS * 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function packUniforms(camera: Camera, lights: PointLight[], materials: Material[], screenWidth: number, screenHeight: number, options: RenderOptions) {
  const data = new Float32Array(UNIFORM_FLOATS);

  // Camera (floats 0–15)
  const basis = getCameraBasis(camera);
  const fovFactor = Math.tan(camera.fov / 2);
  data[0]  = camera.position[0]; data[1]  = camera.position[1]; data[2]  = camera.position[2];
  data[3]  = fovFactor;
  data[4]  = basis.forward[0];   data[5]  = basis.forward[1];   data[6]  = basis.forward[2];
  data[7]  = camera.aspect;
  data[8]  = basis.right[0];     data[9]  = basis.right[1];     data[10] = basis.right[2];
  data[11] = lights.length;
  data[12] = basis.up[0];        data[13] = basis.up[1];        data[14] = basis.up[2];

  // Lights (floats 16–47, 8 floats per light)
  for (let i = 0; i < lights.length; i++) {
    const o = 16 + i * 8;
    data[o]    = lights[i].position[0];  data[o+1]  = lights[i].position[1];  data[o+2]  = lights[i].position[2];
    data[o+3]  = lights[i].intensity;
    data[o+4]  = lights[i].color[0];     data[o+5]  = lights[i].color[1];     data[o+6]  = lights[i].color[2];
  }

  // Screen size (floats 48–51)
  data[48] = screenWidth;
  data[49] = screenHeight;

  // Materials (floats 52–179, 8 floats per material)
  for (let i = 0; i < materials.length; i++) {
    const o = 52 + i * 8;
    data[o]   = materials[i].diffuseAlbedo[0]; data[o+1] = materials[i].diffuseAlbedo[1]; data[o+2] = materials[i].diffuseAlbedo[2];
    data[o+3] = materials[i].roughness || 0;
    data[o+4] = materials[i].fresnel[0];       data[o+5] = materials[i].fresnel[1];       data[o+6] = materials[i].fresnel[2];
    data[o+7] = materials[i].metalness || 0;
  }

  data[180] = floor.height;
  const u32 = new Uint32Array(data.buffer);
  u32[181] = floor.materialId;
  u32[182] = (options.showParticles ? 1 : 0)
    | (options.showTriangles ? 2 : 0)
    | (options.showFloor ? 4 : 0);

  return data;
}

// Sphere layout: center.xyz, radius, materialId(u32), three padding words (32 bytes).
export function packSpheres(spheres: Particle[]) {
  const data = new ArrayBuffer(spheres.length * 32);
  const f32 = new Float32Array(data);
  const u32 = new Uint32Array(data);
  for (let i = 0; i < spheres.length; i++) {
    const o = i * 8;
    f32[o] = spheres[i].center[0]; f32[o+1] = spheres[i].center[1]; f32[o+2] = spheres[i].center[2];
    f32[o+3] = spheres[i].radius;
    u32[o+4] = spheres[i].materialId;
  }
  return data;
}

export function createSphereBuffer(device: GPUDevice, spheres: Particle[]) {
  const data = packSpheres(spheres);
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

// Static particle indices; triangle vertices use the current sphere centers on the GPU.
export function createTriangleBuffer(device: GPUDevice, triangles: Triangle[]) {
  const data = new Uint32Array(triangles.length * 4);
  for (let i = 0; i < triangles.length; i++) {
    const triangle = triangles[i];
    const o = i * 4;
    data[o] = triangle.a;
    data[o+1] = triangle.b;
    data[o+2] = triangle.c;
    data[o+3] = triangle.materialId;
  }
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export function createPipeline(device: GPUDevice, format: GPUTextureFormat, shaderCode: string) {
  const module = device.createShaderModule({ code: shaderCode });
  return device.createRenderPipeline({
    layout: "auto",
    vertex:   { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
  });
}

export function createComputePipeline(device: GPUDevice, _format: GPUTextureFormat, shaderCode: string) {
  const module = device.createShaderModule({ code: shaderCode });
  return device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "computeMain" },
  });
}

export function createComputeBindGroup(device: GPUDevice, pipeline: GPUComputePipeline, uniformBuffer: GPUBuffer, sphereBuffer: GPUBuffer, triangleBuffer: GPUBuffer, accumulationBuffer: GPUBuffer) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: sphereBuffer } },
      { binding: 2, resource: { buffer: triangleBuffer } },
      { binding: 8, resource: { buffer: accumulationBuffer } },
    ],
  });
}

export function createDisplayBindGroup(device: GPUDevice, pipeline: GPURenderPipeline, accumulationBuffer: GPUBuffer, uniformBuffer: GPUBuffer) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: accumulationBuffer } },
      { binding: 1, resource: { buffer: uniformBuffer } },
    ],
  });
}

export function renderFrame(device: GPUDevice, context: GPUCanvasContext, displayPipeline: GPURenderPipeline, computePipeline: GPUComputePipeline, displayBindGroup: GPUBindGroup, computeBindGroup: GPUBindGroup, width: number, height: number) {
  const encoder = device.createCommandEncoder();

  const computePass = encoder.beginComputePass();
  computePass.setPipeline(computePipeline);
  computePass.setBindGroup(0, computeBindGroup);
  computePass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  computePass.end();

  const displayPassDesc: GPURenderPassDescriptor = {
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
    }],
  };

  const displayPass = encoder.beginRenderPass(displayPassDesc);
  displayPass.setPipeline(displayPipeline);
  displayPass.setBindGroup(0, displayBindGroup);
  displayPass.draw(3);
  displayPass.end();

  device.queue.submit([encoder.finish()]);
}
